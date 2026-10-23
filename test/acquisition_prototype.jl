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
sim_ref = CA.get_simulation(config)
sim_obs = CA.get_simulation(config)
int_ref = sim_ref.integrator
stock = sim_obs.integrator
prob = stock.sol.prob
Y = prob.u0
p = prob.p
FT = eltype(parent(Y.c))

components(x) = (getproperty(x, k) for k in propertynames(x))
maxdiff(a, b) = maximum(
    maximum(abs, parent(x) .- parent(y)) for (x, y) in zip(components(a), components(b))
)
maxabs(a) = maximum(maximum(abs, parent(x)) for x in components(a))
exact_equal(a, b) =
    all(parent(x) == parent(y) for (x, y) in zip(components(a), components(b)))
close_enough(a, b; rtol) = all(
    isapprox(parent(x), parent(y); rtol) for (x, y) in zip(components(a), components(b))
)
report(name, value) = println(rpad(name, 56), " = ", value)

@testset "0 runtime assertions" begin
    report("do_dss(axes(Y.c))", CA.do_dss(axes(Y.c)))
    @test CA.do_dss(axes(Y.c))
    qs = Spaces.quadrature_style(Spaces.horizontal_space(axes(Y.c)))
    report("quadrature style", qs)
    @test qs isa Quadratures.GLL{4}
    for (name, expected) in (
        (:turbconv_model, nothing),
        (:rayleigh_sponge, nothing),
        (:viscous_sponge, nothing),
        (:ls_adv, nothing),
    )
        if hasproperty(p.atmos, name)
            value = getproperty(p.atmos, name)
            report("p.atmos.$name", value)
            @test value === expected
        else
            report("p.atmos.$name", "FIELD ABSENT")
            @test_broken hasproperty(p.atmos, name)
        end
    end
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
end

obs = AP.Observer()
int_obs = AP.instrumented_integrator(sim_obs, obs)
for _ in 1:2
    CTS.step!(int_ref)
    CTS.step!(int_obs)
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
    init_events = [e for e in obs.events if e.step == 0]
    report("step-0 events", [(e.hook, e.phase, e.site) for e in init_events])
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
        println(
            "step $step raw sequence: ",
            join(["$(e.stage)/$(e.hook)@$(e.site)" for e in es], " "),
        )
        counts = Dict(
            h => count(e -> e.hook == h, es) for h in (MAPPED_HOOKS..., :T_imp!, :Wfact)
        )
        report("step $step counts", sort(collect(counts); by = first))
        report(
            "step $step cache_imp! outside Newton",
            count(e -> e.hook == :cache_imp! && !newton_internal(e), es),
        )
        for stage in 2:4
            report(
                "step $step stage $stage Newton cache_imp! / T_imp! / Wfact",
                (
                    count(
                        e -> e.stage == stage && e.hook == :cache_imp! && e.site == :newton,
                        es,
                    ),
                    count(e -> e.stage == stage && e.hook == :T_imp!, es),
                    count(e -> e.stage == stage && e.hook == :Wfact, es),
                ),
            )
        end
        report(
            "step $step callback events",
            count(e -> e.hook == :step_end_callback && e.step == step, obs.events),
        )
        observed = [e.hook for e in es if !newton_internal(e)]
        step == 2 && @test observed == expected
        step == 2 && observed != expected && println("observed: ", observed)
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
        d = [
            maximum(abs, parent(x) .- parent(y)) for
            (x, y) in zip(components(exit.data[:state]), components(entry.data[:state]))
        ]
        report("$(entry.stage)/$(entry.hook)@$(entry.site) max|exit-entry| (c, f)", d)
        if entry.hook == :cache_imp! && entry.site in (:L219, :L235)
            # Observed: `set_implicit_precomputed_quantities!` writes `Y.f` at these sites,
            # so the brief's "U unchanged across cache_imp!" expectation does not hold.
            df = parent(exit.data[:state].f) .- parent(entry.data[:state].f)
            report(
                "   Y.f names; max|Δ| bottom face / other faces",
                (
                    propertynames(entry.data[:state].f),
                    maximum(abs, selectdim(df, 1, 1)),
                    maximum(abs, selectdim(df, 1, 2:size(df, 1))),
                ),
            )
            @test d[1] == 0
            @test_broken all(iszero, d)
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
        @. eff = (U_exit - temp) / dtγ
        rel = maxdiff(T_imp_read, eff) / maxabs(eff)
        report("stage $stage dtγ / max rel diff T_imp vs (U-temp)/dtγ", (dtγ, rel))
        @test close_enough(T_imp_read, eff; rtol = 1e-5)
    end
end

@testset "6 final maps and endpoint" begin
    es = entries(2)
    final_lim = only(e for e in es if e.hook == :lim! && e.site == :L87)
    k = findfirst(==(final_lim), obs.events)
    temp_exit = obs.events[k + 1].data[:state]
    u_ref = final_lim.data[:u_ref]
    report("final lim!: max|applied limited contribution|", maxdiff(temp_exit, u_ref))
    report("final lim!: max|exit - entry|", maxdiff(temp_exit, final_lim.data[:state]))
    start = only(e for e in es if e.hook == :step_start)
    dt = FT(float(start.data[:dt]))
    T_exp_all, T_lim_all, T_imp_all, b_exp, b_imp = map(
        k -> final_lim.data[k],
        (:T_exp_all, :T_lim_all, :T_imp_all, :b_exp, :b_imp),
    )
    report(
        "b_exp / b_imp / implicit columns",
        (b_exp, b_imp, sort(collect(keys(T_imp_all)))),
    )
    temp_rebuilt = copy(u_ref)
    for (j, T_lim_j) in T_lim_all
        @. temp_rebuilt += dt * b_exp[j] * T_lim_j
    end
    report(
        "final lim! entry vs u + dt Σ b_exp T_lim (rel)",
        maxdiff(temp_rebuilt, final_lim.data[:state]) / maxabs(temp_rebuilt),
    )
    @test close_enough(final_lim.data[:state], temp_rebuilt; rtol = 1e-5)
    Y_asm = copy(temp_exit)
    for (j, T_exp_j) in T_exp_all
        @. Y_asm += dt * b_exp[j] * T_exp_j
    end
    for (j, T_imp_j) in T_imp_all
        @. Y_asm += dt * b_imp[j] * T_imp_j
    end
    final_dss = only(e for e in es if e.hook == :dss! && e.site == :L98)
    report(
        "final dss! entry vs temp + inc_exp + inc_imp (rel)",
        maxdiff(Y_asm, final_dss.data[:state]) / maxabs(Y_asm),
    )
    @test close_enough(final_dss.data[:state], Y_asm; rtol = 1e-5)
    cache_entry = only(e for e in es if e.hook == :cache! && e.site == :L101)
    kc = findfirst(==(cache_entry), obs.events)
    cache_exit = obs.events[kc + 1].data[:state]
    callback = only(e for e in obs.events if e.step == 2 && e.hook == :step_end_callback)
    report(
        "cache! entry vs exit (c, f)",
        [
            maximum(abs, parent(x) .- parent(y)) for
            (x, y) in zip(components(cache_exit), components(cache_entry.data[:state]))
        ],
    )
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

@testset "3 no aliasing" begin
    cache = int_obs.cache
    live = Any[int_obs.u, cache.U, cache.temp]
    for j in 1:4
        for container in (cache.T_exp, cache.T_lim, cache.T_imp)
            try
                push!(live, container[j])
            catch
            end
        end
    end
    live_parents = [parent(c) for x in live for c in components(x)]
    copies = copied_states()
    report("copied FieldVectors / live arrays", (length(copies), length(live)))
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
