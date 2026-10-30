# Test-only ClimaTimeSteppers solver-event consumer of the standalone acquisition tests;
# `include`d after `import ClimaTimeSteppers as CTS`. Not part of runtests.jl.

"""
    EventRecord

One received observation. `kind` is the name of the CTS event type, or `:hook` for a record
of a fixture-local hook wrapper, whose `site` is then the hook name. `stage` is the
recorder's current stage at receipt. `ctx` holds the scalar event context (`nothing` for a
hook record). `site`, `phase` and `role` are those of a `CTS.Operation` or of the wrapper
and `nothing` for the other event types. `data` holds copies of the event payloads, under
the CTS field names.
"""
struct EventRecord
    kind::Symbol
    stage::Int
    ctx::Any
    site::Union{Symbol, Nothing}
    phase::Union{Symbol, Nothing}
    role::Union{Symbol, Nothing}
    data::NamedTuple
end

"""
    EventRecorder()

Solver observer for `CTS.init(prob, alg; solver_observer)`. Event buffers are borrowed until
the observer returns, so each payload is copied on receipt; `ctx.observer` is not kept.
`stage` is set from the latest CTS event, so hook wrappers can label their evaluations
through `record_hook!`. `StepStart` opens stage 1, whose explicit tendency has no event
of its own; a step-level event has stage 0.
"""
mutable struct EventRecorder
    records::Vector{EventRecord}
    stage::Int
end
EventRecorder() = EventRecorder(EventRecord[], 0)

has_column(a, b, j) = !iszero(b[j]) || any(i -> !iszero(a[i, j]), 1:length(b))
copy_columns(T, a, b) = Dict(j => copy(T[j]) for j in 1:length(b) if has_column(a, b, j))
# Tableau weights are immutable `SparseCoeffs`; arrays are copied.
retained(x) = ismutable(x) ? copy(x) : x

payload(e::CTS.StepStart) = (; u = copy(e.u), e.tableau)
payload(e::CTS.StageBaseline) = (; temp = copy(e.temp))
payload(e::CTS.EffectiveStorage) =
    (; T_imp_i = copy(e.T_imp_i), e.i, U = copy(e.U), temp = copy(e.temp), e.dtγ)
payload(e::CTS.AssemblyInputs) = (;
    u = copy(e.u),
    e.dt,
    T_exp = copy_columns(e.T_exp, e.tableau.a_exp, e.tableau.b_exp),
    T_lim = copy_columns(e.T_lim, e.tableau.a_exp, e.tableau.b_exp),
    T_imp = copy_columns(e.T_imp, e.tableau.a_imp, e.tableau.b_imp),
    b_exp = retained(e.b_exp),
    b_imp = retained(e.b_imp),
)
payload(e::CTS.Operation) =
    (; state = copy(e.state), extra = isnothing(e.extra) ? nothing : copy(e.extra))

labels(event) = (nothing, nothing, nothing)
labels(e::CTS.Operation) = (e.site, e.phase, e.role)

scalar_context(ctx) = (; ctx.step, ctx.t, ctx.dt, ctx.dtγ, ctx.t_exp, ctx.t_imp)

function (rec::EventRecorder)(event)
    rec.stage = event isa CTS.StepStart ? 1 : event.ctx.stage
    kind = nameof(typeof(event))
    ctx = scalar_context(event.ctx)
    push!(rec.records, EventRecord(kind, rec.stage, ctx, labels(event)..., payload(event)))
    return nothing
end

"""
    record_hook!(rec, hook, phase, role, data)

Record one observation of a fixture-local hook wrapper, labelled with the current stage.
`data` holds copies made by the wrapper.
"""
record_hook!(rec, hook, phase, role, data) =
    push!(rec.records, EventRecord(:hook, rec.stage, nothing, hook, phase, role, data))
