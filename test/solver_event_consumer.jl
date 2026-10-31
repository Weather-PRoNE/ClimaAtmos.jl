# Test-only ClimaTimeSteppers solver-event consumer of the standalone acquisition tests;
# `include`d after `import ClimaTimeSteppers as CTS`. Not part of runtests.jl.

"""
    EventRecord

One received observation. `kind` is the name of the CTS event type, or `:hook` for a record
of a `wrap_hooks` wrapper, whose `site` is then the hook name. `stage` is the
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
through `record_hook!`. `StepStart` and the explicit tendency that follows it, which has no
event of its own, are recorded under stage 1; the other step-level events under stage 0.
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

Record one observation of a hook wrapper, labelled with the current stage. `data` holds
copies made by the wrapper.
"""
record_hook!(rec, hook, phase, role, data) =
    push!(rec.records, EventRecord(:hook, rec.stage, nothing, hook, phase, role, data))

# Wrappers of the model hooks CTS does not bracket with an event. Each calls the original
# hook once and records copies under the recorder's current stage.
function recorded_T_exp_T_lim(f!, rec)
    return function (T_exp, T_lim, U, p, t)
        role = :stage_tendency
        record_hook!(rec, :T_exp_T_lim!, :entry, role, (; state = copy(U)))
        result = f!(T_exp, T_lim, U, p, t)
        data = (; state = copy(U), T_exp = copy(T_exp), T_lim = copy(T_lim))
        record_hook!(rec, :T_exp_T_lim!, :exit, role, data)
        return result
    end
end

function recorded_T_imp(f!, rec)
    return function (T, U, p, t)
        role = :provisional_evaluation
        record_hook!(rec, :T_imp!, :entry, role, (; state = copy(U)))
        result = f!(T, U, p, t)
        record_hook!(rec, :T_imp!, :exit, role, (; state = copy(U), output = copy(T)))
        return result
    end
end

function recorded_Wfact(f!, rec)
    return function (W, U, p, dtγ, t)
        role = :jacobian_update
        record_hook!(rec, :Wfact, :entry, role, (; dtγ))
        result = f!(W, U, p, dtγ, t)
        record_hook!(rec, :Wfact, :exit, role, (;))
        return result
    end
end

function recorded_T_post_imp(f!, rec)
    isnothing(f!) && return nothing
    return function (dU, U, p, t)
        role = :correction_scratch
        record_hook!(rec, :T_post_imp!, :entry, role, (; state = copy(U)))
        result = f!(dU, U, p, t)
        record_hook!(rec, :T_post_imp!, :exit, role, (; state = copy(U), output = copy(dU)))
        return result
    end
end

# `wrap_hooks` rebuilds a `ClimaODEFunction` by listing these fields.
const CLIMA_ODE_FUNCTION_FIELDS = (
    :T_exp_T_lim!,
    :T_imp!,
    :T_post_imp!,
    :lim!,
    :dss!,
    :constrain_state!,
    :initialize_imp!,
    :cache!,
    :cache_imp!,
    :update_cache,
    :update_constrain_state,
)

"""
    wrap_hooks(f, rec; record_T_exp_T_lim = true, record_Wfact = true)

Copy of the `ClimaODEFunction` `f` whose `T_imp!` and `T_post_imp!`, and optionally
`T_exp_T_lim!` and `Wfact`, record into `rec`; absent hooks stay `nothing`. Errors if `f`
does not have exactly `CLIMA_ODE_FUNCTION_FIELDS`, since a field left out of the copy would
silently take its default.
"""
function wrap_hooks(f, rec; record_T_exp_T_lim = true, record_Wfact = true)
    fields = fieldnames(typeof(f))
    Set(fields) == Set(CLIMA_ODE_FUNCTION_FIELDS) || error(
        "ClimaODEFunction fields $(fields) differ from the pinned set $(CLIMA_ODE_FUNCTION_FIELDS)",
    )
    T_exp_T_lim! =
        record_T_exp_T_lim ? recorded_T_exp_T_lim(f.T_exp_T_lim!, rec) : f.T_exp_T_lim!
    T_imp! = if isnothing(f.T_imp!)
        nothing
    else
        CTS.ODEFunction(
            recorded_T_imp(f.T_imp!.f, rec);
            jac_prototype = f.T_imp!.jac_prototype,
            Wfact = record_Wfact ? recorded_Wfact(f.T_imp!.Wfact, rec) : f.T_imp!.Wfact,
            tgrad = f.T_imp!.tgrad,
        )
    end
    return CTS.ClimaODEFunction(;
        T_exp_T_lim!,
        T_imp!,
        T_post_imp! = recorded_T_post_imp(f.T_post_imp!, rec),
        f.lim!,
        f.dss!,
        f.constrain_state!,
        f.initialize_imp!,
        f.cache!,
        f.cache_imp!,
        f.update_cache,
        f.update_constrain_state,
    )
end
