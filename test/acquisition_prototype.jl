using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS
import ClimaCore: Spaces, Fields, Quadratures

# Standalone prototype test (B001 T2, local result, prototype); not part of runtests.jl.
const AP = CA.AcquisitionPrototype

config = CA.AtmosConfig(
    Dict(
        "config" => "sphere",
        "initial_condition" => "DryBaroclinicWave",
        "h_elem" => 2,
        "nh_poly" => 3,
        "z_elem" => 10,
        "dt" => "1secs",
        "t_end" => "2secs",
        "FLOAT_TYPE" => "Float32",
        "log_progress" => false,
        "enable_diagnostics" => false,
        "output_default_diagnostics" => false,
        "disable_surface_flux_tendency" => true,
        "output_dir" => mktempdir(),
    );
    job_id = "b001_t2_acquisition",
)
# Construction inside a testset so that a setup error still prints a summary.
@testset "setup" begin
    global sim_ref = CA.get_simulation(config)
    global sim_obs = CA.get_simulation(config)
end
int_ref = sim_ref.integrator
stock = sim_obs.integrator
prob = stock.sol.prob
Y = prob.u0
p = prob.p
FT = eltype(parent(Y.c))

components(x) = (getproperty(x, k) for k in propertynames(x))
exact_equal(a, b) =
    all(parent(x) == parent(y) for (x, y) in zip(components(a), components(b)))
# The prognostic fields of the fixture, compared one at a time: `ρe_tot` is O(10⁵)
# while `ρ` is O(1), so a comparison over the stacked `parent(Y.c)` would hide
# errors in `ρ` and the velocities.
fields(x) = (ρ = x.c.ρ, ρe_tot = x.c.ρe_tot, uₕ = x.c.uₕ, u₃ = x.f.u₃)
# Per field: max |a - b|, and the same normalised by the field's own max |b|.
field_absdiff(a, b) =
    map((x, y) -> maximum(abs, parent(x) .- parent(y)), fields(a), fields(b))
field_reldiff(a, b) = map(
    (x, y) -> maximum(abs, parent(x) .- parent(y)) / maximum(abs, parent(y)),
    fields(a),
    fields(b),
)
report(name, value) = println(rpad(name, 56), " = ", value)
# Max |Δ Y.f.u₃| from `a` to `b` at the bottom face, the top face and the interior faces;
# the first dimension of the parent array indexes the vertical levels.
function face_split(a, b)
    df = parent(b.f.u₃) .- parent(a.f.u₃)
    nv = size(df, 1)
    @assert nv == Spaces.nlevels(axes(Y.f))
    return (
        bottom = maximum(abs, selectdim(df, 1, 1)),
        top = maximum(abs, selectdim(df, 1, nv)),
        interior = maximum(abs, selectdim(df, 1, 2:(nv - 1))),
    )
end

@testset "0 runtime assertions" begin
    report("do_dss(axes(Y.c))", CA.do_dss(axes(Y.c)))
    @test CA.do_dss(axes(Y.c))
    qs = Spaces.quadrature_style(Spaces.horizontal_space(axes(Y.c)))
    report("quadrature style", qs)
    @test qs isa Quadratures.GLL{4}
    # `AtmosModel` forwards these grouped fields through `getproperty`, so
    # `hasproperty` does not see them.
    report("p.atmos.turbconv_model", p.atmos.turbconv_model)
    @test p.atmos.turbconv_model === nothing
    report("p.atmos.rayleigh_sponge", p.atmos.rayleigh_sponge)
    @test p.atmos.rayleigh_sponge === nothing
    report("p.atmos.viscous_sponge", p.atmos.viscous_sponge)
    @test p.atmos.viscous_sponge === nothing
    report("p.atmos.ls_adv", p.atmos.ls_adv)
    @test p.atmos.ls_adv === nothing
    report("p.atmos.microphysics_model", typeof(p.atmos.microphysics_model))
    @test p.atmos.microphysics_model isa CA.DryModel
    user_callbacks = stock.callback.discrete_callbacks[1:(end - 1)]
    report("stock user callbacks", map(cb -> nameof(typeof(cb.affect!)), user_callbacks))
    @test length(user_callbacks) == 2
    report("integrator._dt (type)", (stock._dt, typeof(stock._dt)))
    @test float(stock._dt) == 1
    tab = stock.cache.tableau
    report("alg / tableau a_imp[2,2]", (nameof(typeof(stock.alg.name)), tab.a_imp[2, 2]))
    @test stock.alg.name isa CTS.ARS343
    @test length(tab.b_exp) == 4
    @test tab.a_imp[2, 2] ≈ 0.4358665215084590
    @test stock.alg.tableau.a_imp == tab.a_imp
    @test prob.f.T_post_imp! !== nothing
    report(
        "update_cache / update_constrain_state",
        (prob.f.update_cache, prob.f.update_constrain_state),
    )
    @test prob.f.update_cache isa CTS.UpdateEvery{CTS.EndOfStep}
    @test prob.f.update_constrain_state isa CTS.UpdateEvery{CTS.EndOfStep}
end

obs = AP.Observer()
@testset "setup: two steps" begin
    global int_obs = AP.instrumented_integrator(sim_obs, obs)
    for _ in 1:2
        CTS.step!(int_ref)
        CTS.step!(int_obs)
    end
end

entries(step) = [e for e in obs.events if e.step == step && e.phase == :entry]
pairs_of(step) = [
    (obs.events[k], obs.events[k + 1]) for k in eachindex(obs.events) if
    obs.events[k].step == step && obs.events[k].phase == :entry &&
        obs.events[k].hook != :step_start
]
newton_internal(e) = e.hook in (:T_imp!, :Wfact) || e.site == :newton
const MAPPED_HOOKS = (
    :T_exp_T_lim!,
    :lim!,
    :dss!,
    :initialize_imp!,
    :T_post_imp!,
    :constrain_state!,
    :cache!,
    :cache_imp!,
)

@testset "1 trace" begin
    stage_seq = [
        :lim!,
        :dss!,
        :initialize_imp!,
        :dss!,
        :cache_imp!,
        :cache_imp!,
        :T_post_imp!,
        :dss!,
        :cache_imp!,
        :T_exp_T_lim!,
    ]
    expected = vcat(
        :T_exp_T_lim!,
        repeat(stage_seq, 3),
        [:lim!, :dss!, :constrain_state!, :cache!],
    )
    for step in 1:2
        es = filter(e -> e.hook != :step_start, entries(step))
        observed = [e.hook for e in es if !newton_internal(e)]
        @test observed == expected
        # One Newton iteration per implicit stage: one `T_imp!` and one `Wfact` each,
        # and no `cache_imp!` inside Newton.
        for stage in 2:4
            in_stage(h) = count(e -> e.stage == stage && e.hook == h, es)
            @test (in_stage(:T_imp!), in_stage(:Wfact)) == (1, 1)
            @test count(
                e -> e.stage == stage && e.hook == :cache_imp! && e.site == :newton,
                es,
            ) == 0
        end
        @test count(e -> e.hook == :Wfact, es) == 3
        @test count(e -> e.hook == :step_end_callback && e.step == step, obs.events) == 1
        @test count(e -> e.site == :unexpected, es) == 0
    end
    start = only(e for e in entries(2) if e.hook == :step_start)
    report(
        "step 2 start: U aliases u / U equals u / dt",
        (start.data[:U_aliases_u], start.data[:U_equals_u], start.data[:dt]),
    )
    @test start.data[:U_equals_u]
    lims = [e for e in entries(2) if e.hook == :lim!]
    report(
        "step 2 lim! u_ref === integrator.u",
        [e.data[:u_ref_is_integrator_u] for e in lims],
    )
    @test all(e -> e.data[:u_ref_is_integrator_u], lims)
end

@testset "2 observer independence" begin
    report("t ref / obs", (int_ref.t, int_obs.t))
    @test parent(int_obs.u.c) == parent(int_ref.u.c)
    @test parent(int_obs.u.f) == parent(int_ref.u.f)
    @test int_obs.t == int_ref.t
end

@testset "4 before/after differences in step 2" begin
    for (entry, exit) in pairs_of(2)
        @test exit.hook == entry.hook && exit.phase == :exit
        haskey(entry.data, :state) || continue
        d = field_absdiff(exit.data[:state], entry.data[:state])
        if entry.hook == :cache_imp! && entry.site in (:L219, :L235)
            # `set_implicit_precomputed_quantities!` sets `Y.f.u₃` at the bottom and top
            # faces; D004 classifies these writes as subordinate stage-processing entries.
            Δf = face_split(entry.data[:state], exit.data[:state])
            @test exit.role == entry.role == :subordinate_stage_write
            @test (d.ρ, d.ρe_tot, d.uₕ) == (0, 0, 0)
            @test Δf.bottom > 0
            @test Δf.top > 0
            @test Δf.interior == 0
        elseif entry.hook in (:cache_imp!, :T_post_imp!, :T_exp_T_lim!, :T_imp!)
            @test all(iszero, d)
        end
    end
end

@testset "5 effective implicit identity" begin
    es = entries(2)
    for stage in 2:4
        init = only(e for e in es if e.stage == stage && e.hook == :initialize_imp!)
        temp = init.data[:state]
        dtγ = init.data[:dtγ]
        k = findfirst(
            e ->
                e.step == 2 && e.stage == stage && e.hook == :cache_imp! &&
                e.site == :L253 && e.phase == :exit,
            obs.events,
        )
        U_exit = obs.events[k].data[:state]
        reader =
            only(e for e in es if e.hook == :lim! && get(e.data, :T_imp_index, 0) == stage)
        T_imp_read = reader.data[:T_imp_read]
        eff = similar(temp)
        # The same broadcast that ClimaTimeSteppers uses to set `T_imp[i]`, so every
        # field must show exact numerical equality (zero maximum absolute difference).
        @. eff = (U_exit - temp) / dtγ
        Δ = field_absdiff(T_imp_read, eff)
        @test Δ == map(zero, Δ)
    end
end

@testset "6 final maps and endpoint" begin
    es = entries(2)
    final_lim = only(e for e in es if e.hook == :lim! && e.site == :L87)
    k = findfirst(==(final_lim), obs.events)
    temp_exit = obs.events[k + 1].data[:state]
    u_ref = final_lim.data[:u_ref]
    start = only(e for e in es if e.hook == :step_start)
    dt = FT(float(start.data[:dt]))
    T_exp_all, T_lim_all, T_imp_all, b_exp, b_imp = map(
        k -> final_lim.data[k],
        (:T_exp_all, :T_lim_all, :T_imp_all, :b_exp, :b_imp),
    )
    # Observed as exact numerical equality (zero maximum absolute difference) on this
    # fixture.
    temp_rebuilt = copy(u_ref)
    for (j, T_lim_j) in T_lim_all
        @. temp_rebuilt += dt * b_exp[j] * T_lim_j
    end
    Δ = field_absdiff(final_lim.data[:state], temp_rebuilt)
    @test Δ == map(zero, Δ)
    # The assembly sums the columns in a different order than the solver's fused
    # broadcast, so it matches to rounding only; each field is scaled by its own max.
    Y_asm = copy(temp_exit)
    for (j, T_exp_j) in T_exp_all
        @. Y_asm += dt * b_exp[j] * T_exp_j
    end
    for (j, T_imp_j) in T_imp_all
        @. Y_asm += dt * b_imp[j] * T_imp_j
    end
    final_dss = only(e for e in es if e.hook == :dss! && e.site == :L98)
    rel = field_reldiff(final_dss.data[:state], Y_asm)
    report("final assembly max relative residual", rel)
    @test all(≤(1e-5), rel)
    cache_entry = only(e for e in es if e.hook == :cache! && e.site == :L101)
    kc = findfirst(==(cache_entry), obs.events)
    cache_exit = obs.events[kc + 1].data[:state]
    callback = only(e for e in obs.events if e.step == 2 && e.hook == :step_end_callback)
    # The final `cache!` writes `Y.f.u₃` at the boundary faces: the top face and, by a
    # smaller amount (7.7e-7 vs 1.1e-5 on this fixture), the bottom face.
    Δ = field_absdiff(cache_exit, cache_entry.data[:state])
    @test (Δ.ρ, Δ.ρe_tot, Δ.uₕ) == (0, 0, 0)
    Δf = face_split(cache_entry.data[:state], cache_exit)
    @test Δf.top > 0
    @test Δf.interior == 0
    @test cache_entry.role == obs.events[kc + 1].role == :final_map
    @test exact_equal(cache_exit, callback.data[:u])
    @test exact_equal(callback.data[:u], int_obs.u)
end

@testset "7 declared-inactive checks" begin
    for step in 1:2
        es = entries(step)
        in_stage =
            count(e -> e.hook in (:constrain_state!, :cache!) && !isnothing(e.stage), es)
        final_c = count(e -> e.hook == :constrain_state! && isnothing(e.stage), es)
        final_cache = count(e -> e.hook == :cache! && isnothing(e.stage), es)
        report(
            "step $step in-stage constrain/cache, final constrain, final cache",
            (in_stage, final_c, final_cache),
        )
        @test in_stage == 0
        @test final_c == 1
        @test final_cache == 1
    end
end

copied_states() = [
    x for e in obs.events for v in values(e.data) for
    x in (v isa Dict ? values(v) : (v,)) if x isa Fields.FieldVector
]
checksum(x) = hash(Tuple(Array(parent(c)) for c in components(x)))

# Runs last although numbered 3: it mutates `int_obs.u` and takes a third step, which
# would change the state and the event list that testsets 1, 2 and 4–7 read.
@testset "3 no aliasing" begin
    cache = int_obs.cache
    live = Any[int_obs.u, cache.U, cache.temp]
    # A `SparseContainer` keeps its stored tableau columns in its single field; its type
    # parameter maps tableau columns to positions in it and is not an index list.
    stored(c::CTS.SparseContainer) = getfield(c, only(fieldnames(typeof(c))))
    stored(::Nothing) = ()
    containers = (cache.T_exp, cache.T_lim, cache.T_imp)
    layout = map(c -> (nameof(typeof(c)), length(stored(c))), containers)
    report("T_exp, T_lim, T_imp container type and buffer count", layout)
    @test layout == ((:SparseContainer, 4), (:SparseContainer, 4), (:SparseContainer, 3))
    for container in containers, buffer in stored(container)
        push!(live, buffer)
    end
    live_parents = [parent(c) for x in live for c in components(x)]
    copies = copied_states()
    @test !isempty(copies)
    @test all(x -> all(l -> x !== l, live), copies)
    @test all(x -> all(c -> all(l -> parent(c) !== l, live_parents), components(x)), copies)
    before = map(checksum, copies)
    backup = copy(int_obs.u)
    int_obs.u .+= 1
    @test map(checksum, copies) == before
    int_obs.u .= backup
    CTS.step!(int_obs)
    @test map(checksum, copies) == before
    report("events after extra step", length(obs.events))
end
