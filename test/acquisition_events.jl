using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS
import ClimaCore: Spaces, Fields, Quadratures

# Standalone model integration smoke of the solver events on a dry baroclinic-wave sphere;
# not part of runtests.jl. Needs a ClimaTimeSteppers with labelled solver events.
println("CTS from ", pathof(CTS))

include("acquisition_helpers.jl")
include("solver_event_consumer.jl")
report(name, value) = println(rpad(name, 56), " = ", value)

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
    job_id = "acquisition_events",
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

# The prognostic fields of the fixture, compared one at a time: `ρe_tot` is O(10⁵)
# while `ρ` is O(1), so a comparison over the stacked `parent(Y.c)` would hide
# errors in `ρ` and the velocities.
fields(x) = (ρ = x.c.ρ, ρe_tot = x.c.ρe_tot, uₕ = x.c.uₕ, u₃ = x.f.u₃)
# Per field: max |a - b| normalised by the field's own max |b|.
field_reldiff(a, b) = map(
    (x, y) -> maximum(abs, parent(x) .- parent(y)) / maximum(abs, parent(y)),
    fields(a),
    fields(b),
)

@testset "environment" begin
    @test CTS.supports_solver_events(stock.alg)
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

rec = EventRecorder()
# The stock integrator of `sim_obs` is not stepped: it shares `prob.u0` and `prob.p`.
@testset "setup: two steps" begin
    global int_obs = CTS.init(
        CTS.ODEProblem(wrap_hooks(prob.f, rec), prob.u0, prob.tspan, prob.p),
        stock.alg;
        dt = stock._dt,
        saveat = stock._saveat,
        callback = CTS.CallbackSet(stock.callback.discrete_callbacks[1:(end - 1)]...),
        solver_observer = rec,
    )
    for _ in 1:2
        CTS.step!(int_ref)
        CTS.step!(int_obs)
    end
end

starts = findall(e -> e.kind == :StepStart, rec.records)
step_records(n) =
    rec.records[starts[n]:(n < length(starts) ? starts[n + 1] - 1 : lastindex(rec.records))]
of_kind(es, kind) = [e for e in es if e.kind == kind]
at_site(es, site, phase) = only(e for e in es if e.site == site && e.phase == phase)
hook_entries(es, hook) = [e for e in es if e.site == hook && e.phase == :entry]

const STAGE_SITES = [
    :stage_lim,
    :stage_dss_pre,
    :init_imp,
    :stage_dss_post_init,
    :cache_imp_pre_newton,
    :cache_imp_post_newton,
    :post_imp_correction,
    :T_post_imp!,
    :stage_dss_post,
    :stage_cache_fallback,
    :T_exp_T_lim!,
]
const FINAL_SITES = [:final_lim, :final_dss, :final_constrain, :final_cache]

@testset "endpoint stock vs recorder-observed" begin
    @test length(starts) == 2
    report("t ref / obs", (int_ref.t, int_obs.t))
    report("max|u_obs - u_ref|", maxdiff(int_obs.u, int_ref.u))
    @test exact_equal(int_obs.u, int_ref.u)
    @test int_obs.t == int_ref.t
end

@testset "operation order and event counts" begin
    for step in 1:2
        es = step_records(step)
        # One Newton iteration per implicit stage, so `T_imp!` and `Wfact` are left out.
        observed = [
            e.site for
            e in es if e.phase == :entry && !(e.site in (:T_imp!, :Wfact))
        ]
        @test observed == vcat(:T_exp_T_lim!, repeat(STAGE_SITES, 3), FINAL_SITES)
        @test [e.stage for e in of_kind(es, :StageBaseline)] == 2:4
        @test [e.stage for e in of_kind(es, :EffectiveStorage)] == 2:4
        @test length(of_kind(es, :AssemblyInputs)) == 1
        ops = of_kind(es, :Operation)
        report("step $step Operation records", length(ops))
        @test count(e -> e.phase == :entry, ops) == count(e -> e.phase == :exit, ops)
    end
end

@testset "model hook presence" begin
    for step in 1:2
        es = step_records(step)
        @test [e.stage for e in hook_entries(es, :T_exp_T_lim!)] == 1:4
        for hook in (:T_imp!, :Wfact, :T_post_imp!)
            @test [e.stage for e in hook_entries(es, hook)] == 2:4
        end
        # The hooks write only their outputs.
        for hook in (:T_exp_T_lim!, :T_imp!, :T_post_imp!)
            entries = [e for e in es if e.site == hook && e.phase == :entry]
            exits = [e for e in es if e.site == hook && e.phase == :exit]
            @test length(entries) == length(exits)
            @test all(
                exact_equal(x.data[:state], y.data[:state]) for
                (x, y) in zip(entries, exits)
            )
        end
    end
end

@testset "stage cache writes in step 2" begin
    es = step_records(2)
    # `set_implicit_precomputed_quantities!` sets `Y.f.u₃` at the bottom and top faces.
    for site in (:cache_imp_pre_newton, :cache_imp_post_newton), stage in 2:4
        in_stage = [e for e in es if e.stage == stage]
        entry = at_site(in_stage, site, :entry)
        exit = at_site(in_stage, site, :exit)
        Δf = face_split(entry.data[:state], exit.data[:state])
        @test exit.role == entry.role == :subordinate_stage_write
        @test parent(entry.data[:state].c) == parent(exit.data[:state].c)
        @test Δf.bottom > 0
        @test Δf.top > 0
        @test Δf.interior == 0
    end
    # The post-DSS fallback `cache_imp!` leaves the state unchanged.
    for stage in 2:4
        in_stage = [e for e in es if e.stage == stage]
        entry = at_site(in_stage, :stage_cache_fallback, :entry)
        exit = at_site(in_stage, :stage_cache_fallback, :exit)
        @test exact_equal(entry.data[:state], exit.data[:state])
    end
end

@testset "effective implicit identity in step 2" begin
    es = step_records(2)
    for stage in 2:4
        in_stage = [e for e in es if e.stage == stage]
        stored = only(of_kind(in_stage, :EffectiveStorage))
        baseline = only(of_kind(in_stage, :StageBaseline))
        init = at_site(in_stage, :init_imp, :entry)
        @test stored.data[:i] == stage
        @test exact_equal(baseline.data[:temp], stored.data[:temp])
        @test exact_equal(baseline.data[:temp], init.data[:state])
        @test baseline.ctx.dtγ == stored.data[:dtγ] == init.ctx.dtγ
        eff = similar(stored.data[:temp])
        # The same broadcast that ClimaTimeSteppers uses to set `T_imp[i]`, so every
        # field must show exact numerical equality (zero maximum absolute difference).
        @. eff = (stored.data[:U] - stored.data[:temp]) / stored.data[:dtγ]
        @test maxdiff(stored.data[:T_imp_i], eff) == 0
    end
end

@testset "final maps and endpoint" begin
    es = step_records(2)
    start = only(of_kind(es, :StepStart))
    a = only(of_kind(es, :AssemblyInputs)).data
    dt = FT(float(start.ctx.dt))
    b_exp, b_imp = a[:b_exp], a[:b_imp]
    # The limiter references are the step-start state.
    for site in (:stage_lim, :final_lim), e in es
        e.site == site && e.phase == :entry || continue
        @test exact_equal(e.data[:extra], start.data[:u])
    end
    @test exact_equal(a[:u], start.data[:u])
    final_lim = at_site(es, :final_lim, :entry)
    # Observed as exact numerical equality on this fixture.
    temp_rebuilt = copy(a[:u])
    for (j, T_lim_j) in a[:T_lim]
        @. temp_rebuilt += dt * b_exp[j] * T_lim_j
    end
    @test maxdiff(final_lim.data[:state], temp_rebuilt) == 0
    # The assembly sums the columns in a different order than the solver's fused
    # broadcast, so it matches to rounding only; each field is scaled by its own max.
    Y_asm = copy(at_site(es, :final_lim, :exit).data[:state])
    for (j, T_exp_j) in a[:T_exp]
        @. Y_asm += dt * b_exp[j] * T_exp_j
    end
    for (j, T_imp_j) in a[:T_imp]
        @. Y_asm += dt * b_imp[j] * T_imp_j
    end
    rel = field_reldiff(at_site(es, :final_dss, :entry).data[:state], Y_asm)
    report("final assembly max relative residual", rel)
    @test all(≤(1e-5), rel)
    entry = at_site(es, :final_cache, :entry)
    exit = at_site(es, :final_cache, :exit)
    # The final `cache!` writes `Y.f.u₃` at the boundary faces only.
    Δf = face_split(entry.data[:state], exit.data[:state])
    report("step 2 cache! u₃ split", Δf)
    @test parent(entry.data[:state].c) == parent(exit.data[:state].c)
    @test Δf.top > 0
    @test Δf.interior == 0
    @test entry.role == exit.role == :final_map
    @test exact_equal(exit.data[:state], int_obs.u)
end

copied_states() = [
    x for e in rec.records for v in values(e.data) for
    x in (v isa Dict ? values(v) : (v,)) if x isa Fields.FieldVector
]
checksum(x) = hash(Tuple(Array(parent(c)) for c in components(x)))

# Runs last: it mutates `int_obs.u` and takes a third step.
@testset "retained-copy stability" begin
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
    report(
        "copied FieldVectors / live parent arrays",
        (length(copies), length(live_parents)),
    )
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
    report("records after extra step", length(rec.records))
end
