using Test
import LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS

# Standalone event-channel checks of B001 acceptance claim 7 on the ClimaAtmos side (B001 T9
# part 2, C037 T3); not part of runtests.jl. Needs a ClimaTimeSteppers with labelled solver
# events. The expected event trace for this configuration is specified below; all
# comparisons are exact except the endpoint value (see `endpoint_bound`).
println("CTS from ", pathof(CTS))

include("solver_event_consumer.jl")
report(name, value) = println(rpad(name, 60), " = ", value)

struct ScalarW{FT}
    w::Base.RefValue{FT}
end
Base.similar(W::ScalarW) = ScalarW(Ref(W.w[]))
Base.zero(W::ScalarW) = ScalarW(Ref(zero(W.w[])))
LinearAlgebra.ldiv!(x, W::ScalarW, b) = (x .= b ./ W.w[]; x)

# Native bytes of every array reachable from a solver event's fields.
payload_bytes(x::AbstractArray{<:Number}) = [copy(reinterpret(UInt8, vec(collect(x))))]
payload_bytes(x::Union{Tuple, NamedTuple, AbstractArray, AbstractDict}) =
    reduce(vcat, (payload_bytes(v) for v in values(x)); init = Vector{UInt8}[])
payload_bytes(x) = Vector{UInt8}[]
event_bytes(event) = reduce(
    vcat,
    (payload_bytes(getfield(event, k)) for k in fieldnames(typeof(event)));
    init = Vector{UInt8}[],
)

# Test-only guard around each observer call: borrowed payloads must be byte-identical before
# and after the call. It also keeps the borrowed stage-2 effective-storage `U` and its copy.
struct Guard{O}
    inner::O
    clean::Vector{Bool}
    stage2::Vector{Any}
end
Guard(inner) = Guard(inner, Bool[], Any[])
function (g::Guard)(event)
    before = event_bytes(event)
    g.inner(event)
    push!(g.clean, before == event_bytes(event))
    if event isa CTS.EffectiveStorage && event.ctx.stage == 2
        push!(g.stage2, (event.U, copy(event.U)))
    end
    return nothing
end

# Fault observer: records like the recorder, then changes one supplied state slot by 1.
mutable struct Tamper{O}
    inner::O
    done::Bool
end
function (t::Tamper)(event)
    t.inner(event)
    if !t.done && event isa CTS.Operation && event.state isa AbstractVector
        event.state[1] += 1
        t.done = true
    end
    return nothing
end

# T0 Section 3/7 fixture: u0 = 0, dt = 1, explicit rate 1, implicit rate 2, correction 3,
# with initializer, DSS, constraint and cache hooks present (no-ops on the state). Both update
# policies are step-only, as T0 Section 7 declares; the constructor default `UpdateEvery(
# EndOfStage)` for `cache!` would replace the fallback `cache_imp!` by a stage `cache!`.
function hooks(::Type{FT}) where {FT}
    noop = (Y, p, t) -> nothing
    return CTS.ClimaODEFunction(;
        T_exp_T_lim! = (T_exp, T_lim, U, p, t) ->
            (T_exp .= 1 .+ 0 .* p.sentinel; T_lim .= 0; nothing),
        T_imp! = CTS.ODEFunction(
            (T, U, p, t) -> (T .= 2 .+ 0 .* p.sentinel; nothing);
            jac_prototype = ScalarW(Ref(zero(FT))),
            Wfact = (W, U, p, dtγ, t) -> (W.w[] = -1; nothing),
        ),
        T_post_imp! = (dU, U, p, t) -> (dU .= 3; nothing),
        initialize_imp! = noop,
        dss! = noop,
        constrain_state! = noop,
        cache! = noop,
        cache_imp! = noop,
        update_cache = CTS.UpdateEvery(CTS.EndOfStep),
        update_constrain_state = CTS.UpdateEvery(CTS.EndOfStep),
    )
end

#=
Endpoint bound. In exact arithmetic the step gives u = 0 + 1 + 2 + 3 = 6. In floating point the
ARS343 coefficients are rounded to FT (relative error eps/2 each) and the result is formed from
≤ 4 explicit, 3 implicit and 3 correction contributions through ≤ 4 nested stage sums, each
term of magnitude ≤ 6 · max|coefficient| ≤ 6 · 2. Counting ≤ 32 roundings of relative size
eps/2 on magnitudes ≤ 12 gives ≤ 32 · 12 · eps/2 = 192 eps; we use 256 eps(FT) (Float32:
3.1e-5, Float64: 5.7e-14), well below the fault magnitude 1 of the tamper control.
=#
endpoint_bound(::Type{FT}) where {FT} = 256 * eps(FT)

function guarded_step(::Type{FT}; tamper = false) where {FT}
    recorder = EventRecorder()
    guard = Guard(tamper ? Tamper(recorder, false) : recorder)
    p = (; sentinel = FT[7])
    prob = CTS.ODEProblem(wrap_hooks(hooks(FT), recorder), FT[0], (FT(0), FT(1)), p)
    alg = CTS.IMEXAlgorithm(CTS.ARS343(), CTS.NewtonsMethod(; max_iters = 1))
    integrator = CTS.init(prob, alg; dt = FT(1), solver_observer = guard)
    CTS.step!(integrator)
    return integrator, recorder.records, guard
end

function check_no_write(::Type{FT}) where {FT}
    integrator, events, guard = guarded_step(FT)
    report(
        "C037 claim 7 $FT guarded observer calls (clean, total)",
        (count(guard.clean), length(guard.clean)),
    )
    @test !isempty(guard.clean) && all(guard.clean)
    report(
        "C037 claim 7 $FT endpoint u (value, |u - 6|)",
        (only(integrator.u), abs(only(integrator.u) - 6)),
    )
    @test abs(only(integrator.u) - 6) <= endpoint_bound(FT)
    # Fault: a test observer that changes one supplied slot by exactly 1 is caught at once.
    _, _, guard_f = guarded_step(FT; tamper = true)
    report("C037 claim 7 $FT fault tamper (dirty calls)", count(!, guard_f.clean))
    @test count(!, guard_f.clean) == 1
    # Stage-2 copies survive the later reuse of the borrowed buffer.
    borrowed, snapshot = only(guard.stage2)
    kept = only(e for e in events if e.kind == :EffectiveStorage && e.stage == 2).data[:U]
    report(
        "C037 claim 7 $FT stage-2 U (retained, snapshot, buffer now)",
        (kept, snapshot, borrowed),
    )
    @test kept == snapshot
    @test kept !== borrowed
    @test borrowed != snapshot
end

#=
Complete ordered event template for this configuration: (kind, site, stage, phase, role).
Oracle: C037 T0 Section 7 (docs/migration/runs/20261022T211801Z-C005-pi-test-designer.md:391–420)
and the source-derived table of C037 T3-research Finding 2 (docs/build/c037/t3-research.md),
column "both policies step-only": there the post-DSS fallback `cache_imp!`
(`:stage_cache_fallback`) takes the place of the stage `cache!` (`:stage_cache_post`).
=#
pair(kind, site, stage, role) =
    ((kind, site, stage, :entry, role), (kind, site, stage, :exit, role))
op(site, stage, role) = pair(:Operation, site, stage, role)
hook(site, stage, role) = pair(:hook, site, stage, role)
solver(kind, stage) = (kind, nothing, stage, nothing, nothing)
stage_block(s) = (
    op(:stage_lim, s, :stage_construction)...,
    op(:stage_dss_pre, s, :stage_construction)...,
    solver(:StageBaseline, s),
    op(:init_imp, s, :stage_processing)...,
    op(:stage_dss_post_init, s, :stage_processing)...,
    op(:cache_imp_pre_newton, s, :subordinate_stage_write)...,
    hook(:T_imp!, s, :provisional_evaluation)...,
    hook(:Wfact, s, :jacobian_update)...,
    op(:cache_imp_post_newton, s, :subordinate_stage_write)...,
    (:Operation, :post_imp_correction, s, :entry, :subordinate_stage_write),
    hook(:T_post_imp!, s, :correction_scratch)...,
    (:Operation, :post_imp_correction, s, :exit, :subordinate_stage_write),
    op(:stage_dss_post, s, :stage_processing)...,
    op(:stage_cache_fallback, s, :subordinate_stage_write)...,
    hook(:T_exp_T_lim!, s, :stage_tendency)...,
    solver(:EffectiveStorage, s),
)
const TEMPLATE = [
    solver(:StepStart, 1),
    hook(:T_exp_T_lim!, 1, :stage_tendency)...,
    (x for s in 2:4 for x in stage_block(s))...,
    solver(:AssemblyInputs, 0),
    op(:final_lim, 0, :final_limiter)...,
    (
        x for s in (:final_dss, :final_constrain, :final_cache) for
        x in op(s, 0, :final_map)
    )...,
]
# T0 Section 7 totals: fused explicit 4, cache_imp! 9, dss! 10, T_post_imp! 3,
# initialize_imp! 3, constrain_state! 1, cache! 1 (stage sites as emitted by the fork).
const COUNT_SITES = (
    explicit = (4, (:T_exp_T_lim!,)),
    cache_imp = (9, (:cache_imp_pre_newton, :cache_imp_post_newton, :stage_cache_fallback)),
    dss = (10, (:stage_dss_pre, :stage_dss_post_init, :stage_dss_post, :final_dss)),
    post_imp = (3, (:post_imp_correction,)),
    init_imp = (3, (:init_imp,)),
    constrain = (1, (:final_constrain, :stage_constrain_state)),
    cache = (1, (:final_cache, :stage_cache_post, :stage_cache_no_implicit)),
)

required(events) = [(e.kind, e.site, e.stage, e.phase, e.role) for e in events]
counts(events) = map(
    ((_, sites),) -> count(e -> e.site in sites && e.phase == :entry, events),
    COUNT_SITES,
)
template_accepted(events) =
    required(events) == TEMPLATE && counts(events) == map(first, COUNT_SITES)

function check_template(::Type{FT}) where {FT}
    _, events, _ = guarded_step(FT)
    report("C037 claim 7 $FT counts", counts(events))
    report(
        "C037 claim 7 $FT events (observed, template)",
        (length(events), length(TEMPLATE)),
    )
    k = findfirst(
        i -> i > length(events) || required(events)[i] != TEMPLATE[i],
        eachindex(TEMPLATE),
    )
    report(
        "C037 claim 7 $FT first template mismatch",
        isnothing(k) ? "none" : (k, get(required(events), k, missing), TEMPLATE[k]),
    )
    @test required(events) == TEMPLATE
    @test counts(events) == map(first, COUNT_SITES)
    @test template_accepted(events)
    # Template controls (negative controls of this test's template, not A-side claim-7
    # coverage): remove a baseline, duplicate the final cache pair, swap two final maps.
    find(site, phase; stage = 0) =
        findfirst(e -> e.site == site && e.phase == phase && e.stage == stage, events)
    kb = findfirst(e -> e.kind == :StageBaseline && e.stage == 3, events)
    removed = deleteat!(copy(events), kb)
    kc = find(:final_cache, :entry)
    duplicated = vcat(events[1:(kc + 1)], events[kc:(kc + 1)], events[(kc + 2):end])
    kd, kn = find(:final_dss, :entry), find(:final_constrain, :entry)
    swapped = copy(events)
    swapped[kd:(kd + 1)], swapped[kn:(kn + 1)] = events[kn:(kn + 1)], events[kd:(kd + 1)]
    controls = (removed = removed, duplicated = duplicated, swapped = swapped)
    report("C037 claim 7 $FT template controls accepted", map(template_accepted, controls))
    @test !any(map(template_accepted, controls))
end

@testset "C037 event integrity" begin
    for FT in (Float32, Float64)
        @testset "claim 7 observer no-write $FT" check_no_write(FT)
        @testset "claim 7 lifecycle template $FT" check_template(FT)
    end
end

@testset "wrap_hooks field guard" begin
    @test_throws ErrorException wrap_hooks((; T_imp! = nothing), EventRecorder())
end
