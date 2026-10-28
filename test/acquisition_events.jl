using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS
import ClimaCore: Spaces, Fields

# Standalone equivalence test of the prototype reads and the solver events (B001 T9 part 1,
# local result, prototype); not part of runtests.jl. Needs a ClimaTimeSteppers with
# labelled solver events.
const AP = CA.AcquisitionPrototype
println("CTS from ", pathof(CTS))

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
    job_id = "b001_t9_acquisition_events",
)
sim_ref = CA.get_simulation(config)
sim_proto = CA.get_simulation(config)
sim_ev = CA.get_simulation(config)
int_ref = sim_ref.integrator
stock = sim_ev.integrator
Y = stock.sol.prob.u0

components(x) = (getproperty(x, k) for k in propertynames(x))
maxdiff(a, b) = maximum(
    maximum(abs, parent(x) .- parent(y)) for (x, y) in zip(components(a), components(b))
)
exact_equal(a, b) =
    all(parent(x) == parent(y) for (x, y) in zip(components(a), components(b)))
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

@testset "a environment" begin
    @test CTS.supports_solver_events(stock.alg)
end

obs = AP.Observer()
int_proto = AP.instrumented_integrator(sim_proto, obs)
eobs = AP.EventObserver()
int_ev = AP.instrumented_integrator_events(sim_ev, eobs)
ev = eobs.observer
for _ in 1:2
    CTS.step!(int_ref)
    CTS.step!(int_proto)
    CTS.step!(int_ev)
end

const LABELS = Set(v.label for v in values(AP.SITE_MAP))
label(e) = AP.SITE_MAP[e.site].label
proto_ops = [e for e in obs.events if e.step >= 1 && e.site in LABELS]
event_ops = [e for e in ev.events if haskey(AP.SITE_MAP, e.site)]
key(e) = (e.step, e.stage, e.phase)
in_step(es, step, hook) = [e for e in es if e.step == step && e.hook == hook]

@testset "e end states" begin
    report("t ref / prototype / events", (int_ref.t, int_proto.t, int_ev.t))
    report(
        "max|u_proto - u_ref| / max|u_ev - u_ref|",
        (maxdiff(int_proto.u, int_ref.u), maxdiff(int_ev.u, int_ref.u)),
    )
    @test exact_equal(int_proto.u, int_ref.u)
    @test exact_equal(int_ev.u, int_ref.u)
    @test int_proto.t == int_ref.t == int_ev.t
end

@testset "d counts per step" begin
    for step in 1:2
        es = [e for e in ev.events if e.step == step]
        n(hook) = count(e -> e.hook == hook, es)
        report(
            "step $step StepStart/StageBaseline/EffectiveStorage/Assembly",
            (n(:step_start), n(:baseline), n(:effective_storage), n(:assembly_inputs)),
        )
        @test n(:step_start) == 1
        @test [e.stage for e in in_step(es, step, :baseline)] == 2:4
        @test [e.stage for e in in_step(es, step, :effective_storage)] == 2:4
        @test n(:assembly_inputs) == 1
        ops = [
            e for e in es if
            e.phase == :entry && (haskey(AP.SITE_MAP, e.site) || e.hook == :operation)
        ]
        sites = unique(e.site for e in ops)
        counts = [site => count(e -> e.site == site, ops) for site in sites]
        report("step $step Operation entries per site", counts)
        unmapped = [e.site for e in es if e.hook == :operation]
        isempty(unmapped) || report("step $step unmapped sites", unique(unmapped))
        @test isempty(unmapped)
        @test Set(sites) == Set(keys(AP.SITE_MAP))
        for (site, n_site) in counts
            n_proto = count(
                e ->
                    e.step == step && e.phase == :entry &&
                    e.site == AP.SITE_MAP[site].label,
                obs.events,
            )
            @test n_site == n_proto
            @test n_site == (startswith(string(site), "final_") ? 1 : 3)
        end
        @test count(e -> e.phase == :exit, ops) == 0
        @test count(e -> e.phase == :exit && haskey(AP.SITE_MAP, e.site), es) == length(ops)
    end
end

@testset "c equivalence" begin
    report("labelled events prototype / events", (length(proto_ops), length(event_ops)))
    @test [(key(e)..., e.site, e.hook) for e in proto_ops] ==
          [(key(e)..., label(e), e.hook) for e in event_ops]
    # The `:post_imp_correction` exit follows the application of the correction, which the
    # prototype first sees at the entry of the next `dss!` (`:L245`).
    correction_exit(e) = e.site == :post_imp_correction && e.phase == :exit
    mismatched = [
        (key(e)..., e.site, maxdiff(p.data[:state], e.data[:state])) for
        (p, e) in zip(proto_ops, event_ops) if !exact_equal(p.data[:state], e.data[:state])
    ]
    report("state mismatches (step, stage, phase, site, maxdiff)", mismatched)
    @test all(m -> m[4] == :post_imp_correction && m[3] == :exit, mismatched)
    @test count(e -> !correction_exit(e), event_ops) ==
          length(event_ops) - length(mismatched)
    for (k, e) in enumerate(event_ops)
        correction_exit(e) || continue
        next = proto_ops[k + 1]
        @test (key(next)..., next.site) == (e.step, e.stage, :entry, :L245)
        @test exact_equal(e.data[:state], next.data[:state])
    end
    for (p, e) in zip(proto_ops, event_ops)
        p.hook == :lim! && p.phase == :entry &&
            @test exact_equal(p.data[:u_ref], e.data[:extra])
        p.hook == :initialize_imp! && @test p.data[:dtγ] == e.data[:dtγ]
    end
    for step in 1:2
        for stage in 2:4
            read = only(
                e for e in obs.events if e.step == step && e.hook == :lim! &&
                    e.phase == :entry && get(e.data, :T_imp_index, 0) == stage
            )
            stored = only(
                e for e in in_step(ev.events, step, :effective_storage) if e.stage == stage
            )
            baseline =
                only(e for e in in_step(ev.events, step, :baseline) if e.stage == stage)
            init = only(
                e for e in in_step(ev.events, step, :initialize_imp!) if
                e.stage == stage && e.phase == :entry
            )
            @test stored.data[:T_imp_index] == stage
            @test exact_equal(read.data[:T_imp_read], stored.data[:T_imp_read])
            @test exact_equal(baseline.data[:temp], stored.data[:temp])
            @test exact_equal(baseline.data[:temp], init.data[:state])
            @test baseline.data[:dtγ] == stored.data[:dtγ] == init.data[:dtγ]
        end
        final_lim = only(
            e for
            e in in_step(obs.events, step, :lim!) if e.site == :L87 && e.phase == :entry
        )
        assembly = only(in_step(ev.events, step, :assembly_inputs))
        for k in (:T_exp_all, :T_lim_all, :T_imp_all)
            report("step $step $k columns", sort(collect(keys(assembly.data[k]))))
            @test keys(final_lim.data[k]) == keys(assembly.data[k])
            @test all(
                j -> exact_equal(final_lim.data[k][j], assembly.data[k][j]),
                keys(final_lim.data[k]),
            )
        end
        @test final_lim.data[:b_exp] == assembly.data[:b_exp]
        @test final_lim.data[:b_imp] == assembly.data[:b_imp]
        @test exact_equal(final_lim.data[:u_ref], assembly.data[:u])
        proto_start = only(in_step(obs.events, step, :step_start))
        event_start = only(in_step(ev.events, step, :step_start))
        @test exact_equal(proto_start.data[:u], event_start.data[:u])
        @test proto_start.data[:dt] == event_start.data[:dt]
        @test proto_start.data[:integrator_step] == event_start.data[:integrator_step]
        @test proto_start.t == event_start.t
    end
    tableau = only(e for e in obs.events if haskey(e.data, :tableau)).data[:tableau]
    @test all(e -> e.data[:tableau] == tableau, in_step(ev.events, 1, :step_start))
    @test all(e -> e.data[:tableau] == tableau, in_step(ev.events, 2, :step_start))
    # The model hooks are wrapped on both paths and see the same states.
    hooks(es) = [
        (key(e)..., e.hook, e.site) for e in es if
        e.step >= 1 && e.hook in (:T_exp_T_lim!, :T_imp!, :Wfact, :T_post_imp!) &&
            !haskey(AP.SITE_MAP, e.site)
    ]
    @test hooks(obs.events) == hooks(ev.events)
end

@testset "g final-map face split" begin
    for step in 1:2
        pk = findfirst(
            e -> e.step == step && e.site == :L101 && e.phase == :entry,
            obs.events,
        )
        ek = findfirst(
            e -> e.step == step && e.site == :final_cache && e.phase == :entry,
            ev.events,
        )
        @test (ev.events[ek + 1].site, ev.events[ek + 1].phase) == (:final_cache, :exit)
        proto_split =
            face_split(obs.events[pk].data[:state], obs.events[pk + 1].data[:state])
        event_split = face_split(ev.events[ek].data[:state], ev.events[ek + 1].data[:state])
        report("step $step cache! u₃ split prototype", proto_split)
        report("step $step cache! u₃ split events", event_split)
        @test proto_split == event_split
    end
end

copied_states() = [
    x for e in ev.events for v in values(e.data) for
    x in (v isa Dict ? values(v) : (v,)) if x isa Fields.FieldVector
]
checksum(x) = hash(Tuple(Array(parent(c)) for c in components(x)))

@testset "f no aliasing" begin
    cache = int_ev.cache
    live = Any[int_ev.u, cache.U, cache.temp]
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
    report(
        "copied FieldVectors / live parent arrays",
        (length(copies), length(live_parents)),
    )
    @test all(x -> all(l -> x !== l, live), copies)
    @test all(x -> all(c -> all(l -> parent(c) !== l, live_parents), components(x)), copies)
    before = map(checksum, copies)
    backup = copy(int_ev.u)
    int_ev.u .+= 1
    @test map(checksum, copies) == before
    int_ev.u .= backup
    CTS.step!(int_ev)
    @test map(checksum, copies) == before
    report("events after extra step", length(ev.events))
end
