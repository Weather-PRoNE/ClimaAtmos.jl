"""
    AcquisitionPrototype

Prototype acquisition layer (local result, prototype; B001 T2). It observes IMEX ARK steps
of a ClimaAtmos simulation by wrapping the hooks of the `ClimaTimeSteppers.ClimaODEFunction`
that `args_integrator` wires, and by reading pinned ClimaTimeSteppers 1.0.1 cache fields
(`integrator.cache.T_exp`, `T_lim`, `T_imp`, `tableau`) where no hook exists. Every recorded
state is a `copy`; no event holds a reference to a solver array.

Not part of the public API; nothing is exported.
"""
module AcquisitionPrototype

import ClimaTimeSteppers as CTS
import ClimaCore: Fields

"""
    Event

One observation. `stage` is 1–4 inside a stage and `nothing` for the final maps, the
callbacks and initialization; `point` is the row of the B001 acquisition table (0 if not
mapped); `site` names the pinned ClimaTimeSteppers 1.0.1 call site (`:L160` is
`imex_ark.jl:160`, `:newton` is a call inside the Newton solve); `role` follows D004.
"""
struct Event
    step::Int
    stage::Union{Int, Nothing}
    point::Int
    hook::Symbol
    phase::Symbol
    site::Symbol
    role::Symbol
    t::Any
    data::Dict{Symbol, Any}
end

"""
    Observer()

Mutable container of `Event`s plus the stage bookkeeping state. `integrator` is set by
`instrumented_integrator` and is read for `u`, `t`, `dt`, `step` and the cache fields.
"""
mutable struct Observer
    events::Vector{Event}
    integrator::Any
    in_step::Bool
    step::Int
    stage::Union{Int, Nothing}
    n_lim::Int
    stage_phase::Symbol
    n_stages::Int
end
Observer() = Observer(Event[], nothing, false, 0, nothing, 0, :outside, 0)

record!(obs, hook, phase, site, point, role, t, data) = push!(
    obs.events,
    Event(obs.step, obs.stage, point, hook, phase, site, role, t, data),
)

has_column(a, b, j) = !iszero(b[j]) || any(i -> !iszero(a[i, j]), 1:length(b))

# The tableau coefficients are immutable `SparseCoeffs` values, so storing them copies them.
copy_tableau(tab) = Dict{Symbol, Any}(
    name => getproperty(tab, name) for
    name in (:a_exp, :b_exp, :a_imp, :b_imp, :c_exp, :c_imp)
)

# Stage bookkeeping. For ARS343 with the ClimaAtmos defaults (`update_cache_every` and
# `update_constrain_state_every` both `step`), pinned 1.0.1 calls the hooks in this order:
#   stage 1:     T_exp_T_lim!
#   stages 2–4:  lim! (:160), dss! (:202), initialize_imp! (:215), dss! (:216),
#                cache_imp! (:219), Newton block (T_imp!, Wfact, cache_imp! callbacks),
#                cache_imp! (:235), T_post_imp! (:236), dss! (:245), cache_imp! (:253),
#                T_exp_T_lim! (:273)
#   final:       lim! (:87), dss! (:98), constrain_state! (:100), cache! (:101)
#   callbacks:   user callbacks (the observer's step-end callback last), saving callback
# A step begins at the first `T_exp_T_lim!` entry after a step-end callback. The first
# `n_stages - 1` `lim!` entries of a step begin stages 2..n_stages; the next one is the
# final limiter. `stage_phase` tracks the position inside a stage so that each `dss!` and
# `cache_imp!` call is mapped to its call site. The `cache_imp!` call at :235 cannot be told
# apart from a Newton-internal callback when it happens, so the last `cache_imp!` pair
# before `T_post_imp!` is relabelled from `:newton` to `:L235` at the `T_post_imp!` entry.

function begin_step!(obs, U)
    integrator = obs.integrator
    tab = integrator.cache.tableau
    obs.in_step = true
    obs.step += 1
    obs.stage = 1
    obs.n_lim = 0
    obs.n_stages = length(tab.b_exp)
    obs.stage_phase = :stage1
    data = Dict{Symbol, Any}(
        :u => copy(integrator.u),
        :dt => integrator.dt,
        :integrator_step => integrator.step,
        :U_aliases_u => U === integrator.u,
        :U_equals_u => same_values(U, integrator.u),
    )
    if !any(e -> e.point == 1 && haskey(e.data, :tableau), obs.events)
        data[:tableau] = copy_tableau(tab)
    end
    record!(obs, :step_start, :entry, :L70, 1, :step_start, integrator.t, data)
end

same_values(a, b) =
    all(parent(getproperty(a, k)) == parent(getproperty(b, k)) for k in propertynames(a))

state(U) = Dict{Symbol, Any}(:state => copy(U))

function wrap_T_exp_T_lim(f!, obs)
    isnothing(f!) && return nothing
    return function (T_exp_i, T_lim_i, U, p, t)
        if obs.in_step
            obs.stage_phase = :evaluated
            site, point = :L273, 11
        elseif !isnothing(obs.integrator)
            begin_step!(obs, U)
            site, point = :L273, 2
        else
            site, point = :outside_step, 0
        end
        record!(obs, :T_exp_T_lim!, :entry, site, point, :stage_tendency, t, state(U))
        result = f!(T_exp_i, T_lim_i, U, p, t)
        data = state(U)
        data[:T_exp] = copy(T_exp_i)
        data[:T_lim] = copy(T_lim_i)
        record!(obs, :T_exp_T_lim!, :exit, site, point, :stage_tendency, t, data)
        return result
    end
end

function wrap_lim(f!, obs)
    isnothing(f!) && return nothing
    return function (U, p, t, u_ref)
        data = state(U)
        data[:u_ref] = copy(u_ref)
        if obs.in_step
            integrator = obs.integrator
            cache = integrator.cache
            tab = cache.tableau
            data[:u_ref_is_integrator_u] = u_ref === integrator.u
            obs.n_lim += 1
            # point 12: the effective implicit tendency of the previous stage, written at
            # imex_ark.jl:284 without a hook, is read here before it can be used again.
            prev = obs.n_lim
            if has_column(tab.a_imp, tab.b_imp, prev)
                data[:T_imp_read] = copy(cache.T_imp[prev])
                data[:T_imp_index] = prev
            end
            if obs.n_lim < obs.n_stages
                obs.stage = obs.n_lim + 1
                obs.stage_phase = :constructed
                site, point, role = :L160, 3, :stage_construction
            else
                obs.stage = nothing
                obs.stage_phase = :final
                site, point, role = :L87, 14, :final_assembly
                # point 13: channel arrays and weights of the final assembly
                s = obs.n_stages
                data[:T_exp_all] = Dict(
                    j => copy(cache.T_exp[j]) for
                    j in 1:s if has_column(tab.a_exp, tab.b_exp, j)
                )
                data[:T_lim_all] = Dict(
                    j => copy(cache.T_lim[j]) for
                    j in 1:s if has_column(tab.a_exp, tab.b_exp, j)
                )
                data[:T_imp_all] = Dict(
                    j => copy(cache.T_imp[j]) for
                    j in 1:s if has_column(tab.a_imp, tab.b_imp, j)
                )
                data[:b_exp] = tab.b_exp
                data[:b_imp] = tab.b_imp
            end
        else
            site, point, role = :outside_step, 0, :initialization
        end
        record!(obs, :lim!, :entry, site, point, role, t, data)
        result = f!(U, p, t, u_ref)
        record!(obs, :lim!, :exit, site, point, role, t, state(U))
        return result
    end
end

function classify_dss!(obs)
    obs.in_step || return (:outside_step, 0, 0, :initialization)
    isnothing(obs.stage) && return (:L98, 15, 16, :final_map)
    phase = obs.stage_phase
    if phase == :constructed
        obs.stage_phase = :pre_snapshot_dss
        return (:L202, 4, 4, :stage_dss)
    elseif phase == :initialized
        obs.stage_phase = :init_dss
        return (:L216, 6, 6, :stage_dss)
    elseif phase == :post_imp
        obs.stage_phase = :post_solve_dss
        return (:L245, 9, 10, :stage_dss)
    else
        return (:unexpected, 0, 0, :stage_dss)
    end
end

function wrap_dss(f!, obs)
    isnothing(f!) && return nothing
    return function (U, p, t)
        site, point_entry, point_exit, role = classify_dss!(obs)
        record!(obs, :dss!, :entry, site, point_entry, role, t, state(U))
        result = f!(U, p, t)
        record!(obs, :dss!, :exit, site, point_exit, role, t, state(U))
        return result
    end
end

function wrap_constrain_state(f!, obs)
    isnothing(f!) && return nothing
    return function (U, p, t)
        site, point, role = if !obs.in_step
            (:outside_step, 0, :initialization)
        elseif isnothing(obs.stage)
            (:L100, 17, :final_map)
        else
            (:stage_constrain_state, 10, :subordinate_stage_write)
        end
        record!(obs, :constrain_state!, :entry, site, point, role, t, state(U))
        result = f!(U, p, t)
        record!(obs, :constrain_state!, :exit, site, point, role, t, state(U))
        return result
    end
end

function wrap_cache(f!, obs)
    isnothing(f!) && return nothing
    return function (U, p, t)
        site, point, role = if !obs.in_step
            (:init, 0, :initialization)
        elseif isnothing(obs.stage)
            (:L101, 18, :final_map)
        else
            (:stage_cache, 10, :subordinate_stage_write)
        end
        record!(obs, :cache!, :entry, site, point, role, t, state(U))
        result = f!(U, p, t)
        record!(obs, :cache!, :exit, site, point, role, t, state(U))
        return result
    end
end

function wrap_cache_imp(f!, obs)
    isnothing(f!) && return nothing
    return function (U, p, t)
        phase = obs.stage_phase
        site, point = if !obs.in_step
            (:outside_step, 0)
        elseif phase == :init_dss
            obs.stage_phase = :newton
            (:L219, 6)
        elseif phase == :newton
            (:newton, 0)
        elseif phase == :post_solve_dss
            obs.stage_phase = :post_solve_cache
            (:L253, 10)
        else
            (:unexpected, 0)
        end
        role = :subordinate_stage_write
        record!(obs, :cache_imp!, :entry, site, point, role, t, state(U))
        result = f!(U, p, t)
        record!(obs, :cache_imp!, :exit, site, point, role, t, state(U))
        return result
    end
end

function wrap_initialize_imp(f!, obs)
    isnothing(f!) && return nothing
    return function (U, p, dtγ)
        obs.in_step && (obs.stage_phase = :initialized)
        t = isnothing(obs.integrator) ? nothing : obs.integrator.t
        data = state(U)
        data[:dtγ] = dtγ
        record!(obs, :initialize_imp!, :entry, :L215, 5, :implicit_setup, t, data)
        result = f!(U, p, dtγ)
        data = state(U)
        data[:dtγ] = dtγ
        record!(obs, :initialize_imp!, :exit, :L215, 6, :implicit_setup, t, data)
        return result
    end
end

function wrap_T_imp(f!, obs)
    return function (T_imp_i, U, p, t)
        site = obs.stage_phase == :newton ? :L311 : :unexpected
        role = :provisional_evaluation
        record!(obs, :T_imp!, :entry, site, 7, role, t, state(U))
        result = f!(T_imp_i, U, p, t)
        data = state(U)
        data[:output] = copy(T_imp_i)
        record!(obs, :T_imp!, :exit, site, 7, role, t, data)
        return result
    end
end

function wrap_Wfact(f!, obs)
    isnothing(f!) && return nothing
    return function (jacobian, U, p, dtγ, t)
        data = Dict{Symbol, Any}(:dtγ => dtγ)
        record!(obs, :Wfact, :entry, :newton, 0, :jacobian_update, t, data)
        result = f!(jacobian, U, p, dtγ, t)
        record!(obs, :Wfact, :exit, :newton, 0, :jacobian_update, t, Dict{Symbol, Any}())
        return result
    end
end

function relabel(e::Event; point, site)
    return Event(e.step, e.stage, point, e.hook, e.phase, site, e.role, e.t, e.data)
end

function wrap_T_post_imp(f!, obs)
    isnothing(f!) && return nothing
    return function (dU, U, p, t)
        if obs.in_step && obs.stage_phase == :newton
            k = findlast(e -> e.hook == :cache_imp! && e.site == :newton, obs.events)
            if !isnothing(k) && k == lastindex(obs.events)
                obs.events[k - 1] = relabel(obs.events[k - 1]; point = 8, site = :L235)
                obs.events[k] = relabel(obs.events[k]; point = 8, site = :L235)
            end
        end
        obs.in_step && (obs.stage_phase = :post_imp)
        role = :correction_scratch
        record!(obs, :T_post_imp!, :entry, :L236, 9, role, t, state(U))
        result = f!(dU, U, p, t)
        data = state(U)
        data[:output] = copy(dU)
        record!(obs, :T_post_imp!, :exit, :L236, 9, role, t, data)
        return result
    end
end

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
    wrap(f::CTS.ClimaODEFunction, obs::Observer)

Return a new `ClimaODEFunction` whose hooks call the hooks of `f` and record entry and exit
events in `obs`. `nothing` hooks stay `nothing`; the update-signal policies are passed
through unchanged.
"""
function wrap(f::CTS.ClimaODEFunction, obs::Observer)
    fields = fieldnames(typeof(f))
    Set(fields) == Set(CLIMA_ODE_FUNCTION_FIELDS) || error(
        "ClimaODEFunction fields $(fields) differ from the pinned 1.0.1 set $(CLIMA_ODE_FUNCTION_FIELDS)",
    )
    T_imp! = if isnothing(f.T_imp!)
        nothing
    else
        CTS.ODEFunction(
            wrap_T_imp(f.T_imp!.f, obs);
            jac_prototype = f.T_imp!.jac_prototype,
            Wfact = wrap_Wfact(f.T_imp!.Wfact, obs),
            tgrad = f.T_imp!.tgrad,
        )
    end
    return CTS.ClimaODEFunction(;
        T_exp_T_lim! = wrap_T_exp_T_lim(f.T_exp_T_lim!, obs),
        T_imp!,
        T_post_imp! = wrap_T_post_imp(f.T_post_imp!, obs),
        lim! = wrap_lim(f.lim!, obs),
        dss! = wrap_dss(f.dss!, obs),
        constrain_state! = wrap_constrain_state(f.constrain_state!, obs),
        initialize_imp! = wrap_initialize_imp(f.initialize_imp!, obs),
        cache! = wrap_cache(f.cache!, obs),
        cache_imp! = wrap_cache_imp(f.cache_imp!, obs),
        update_cache = f.update_cache,
        update_constrain_state = f.update_constrain_state,
    )
end

"""
    record_step_end!(obs, integrator)

Affect of the observer's step-end callback (point 19): copy `integrator.u` and close the
current step.
"""
function record_step_end!(obs::Observer, integrator)
    obs.stage = nothing
    data = Dict{Symbol, Any}(:u => copy(integrator.u), :integrator_step => integrator.step)
    record!(obs, :step_end_callback, :exit, :callback, 19, :callback, integrator.t, data)
    obs.in_step = false
    obs.stage_phase = :outside
    return nothing
end

"""
    instrumented_integrator(sim, obs; callbacks)

Build a new integrator for the problem of the `AtmosSimulation` `sim`, with its
`ClimaODEFunction` wrapped by `wrap(_, obs)`. `callbacks` defaults to the user callbacks of
`sim.integrator` (all discrete callbacks except the saving callback that `CTS.init` appends
last); the observer's step-end callback is appended after them. The stock integrator of
`sim` must not be stepped, since both integrators share `prob.u0` and `prob.p`.
"""
function instrumented_integrator(
    sim,
    obs::Observer;
    callbacks = sim.integrator.callback.discrete_callbacks[1:(end - 1)],
)
    stock = sim.integrator
    prob = stock.sol.prob
    step_end_callback = CTS.DiscreteCallback(
        (u, t, integrator) -> true,
        integrator -> record_step_end!(obs, integrator),
    )
    integrator = CTS.init(
        CTS.ODEProblem(wrap(prob.f, obs), prob.u0, prob.tspan, prob.p),
        stock.alg;
        dt = stock._dt,
        saveat = stock._saveat,
        callback = CTS.CallbackSet(callbacks..., step_end_callback),
    )
    obs.integrator = integrator
    return integrator
end

# Event path (B001 T9): a ClimaTimeSteppers with labelled solver events reports the solver
# internals that the path above reads from pinned cache fields and reconstructs with the
# stage state machine. Only the model hooks are wrapped; everything else arrives as events.
@static if isdefined(CTS, :supports_solver_events)

    """
        SITE_MAP

    Solver-event sites (`CTS.Operation` `site`) mapped to the pinned 1.0.1 call-site label of
    the prototype path, the hook called there and the B001 table rows of the entry and exit
    records (local result, prototype; B001 T9). Sites outside the map do not execute with the
    ClimaAtmos defaults; their events are recorded under their own name with hook `:operation`.
    """
    const SITE_MAP = Dict{
        Symbol,
        @NamedTuple{label::Symbol, hook::Symbol, points::Tuple{Int, Int}}
    }(
        :stage_lim => (label = :L160, hook = :lim!, points = (3, 3)),
        :stage_dss_pre => (label = :L202, hook = :dss!, points = (4, 4)),
        :init_imp => (label = :L215, hook = :initialize_imp!, points = (5, 6)),
        :stage_dss_post_init => (label = :L216, hook = :dss!, points = (6, 6)),
        :cache_imp_pre_newton => (label = :L219, hook = :cache_imp!, points = (6, 6)),
        :cache_imp_post_newton => (label = :L235, hook = :cache_imp!, points = (8, 8)),
        :post_imp_correction => (label = :L236, hook = :T_post_imp!, points = (9, 9)),
        :stage_dss_post => (label = :L245, hook = :dss!, points = (9, 10)),
        :stage_cache_fallback => (label = :L253, hook = :cache_imp!, points = (10, 10)),
        :final_lim => (label = :L87, hook = :lim!, points = (14, 14)),
        :final_dss => (label = :L98, hook = :dss!, points = (15, 16)),
        :final_constrain =>
            (label = :L100, hook = :constrain_state!, points = (17, 17)),
        :final_cache => (label = :L101, hook = :cache!, points = (18, 18)),
    )

    """
        EventObserver(observer = Observer())

    Solver observer for `CTS.init(prob, alg; solver_observer)` (local result, prototype;
    B001 T9). Called with each solver event, it copies the borrowed payloads at once and
    records `Event`s in `observer`: `StepStart` gives point 1, `StageBaseline` the implicit
    baseline `temp` (site `:baseline`), `EffectiveStorage` point 12, `AssemblyInputs`
    point 13 and `Operation` the entry and exit records of the call sites in `SITE_MAP`.
    `Event.step` counts the steps seen by the observer, `Event.stage` is `ctx.stage`
    (`nothing` for step-level events) and `Event.t` is the step-start time `ctx.t`; the
    stage times of an `Operation` are in `data[:t_exp]` and `data[:t_imp]`. The observer
    never reads the integrator or its cache.
    """
    struct EventObserver
        observer::Observer
    end
    EventObserver() = EventObserver(Observer())

    stage_of(ctx) = iszero(ctx.stage) ? nothing : ctx.stage

    # Tableau weights are immutable `SparseCoeffs`; arrays are copied.
    snapshot(x) = ismutable(x) ? copy(x) : x

    copy_columns(T, a, b) =
        Dict(j => copy(T[j]) for j in 1:length(b) if has_column(a, b, j))

    function (o::EventObserver)(event::CTS.StepStart)
        obs = o.observer
        ctx = event.ctx
        obs.in_step = true
        obs.step += 1
        obs.stage = 1
        obs.n_stages = length(event.tableau.b_exp)
        data = Dict{Symbol, Any}(
            :u => copy(event.u),
            :dt => ctx.dt,
            :integrator_step => ctx.step,
            :tableau => copy_tableau(event.tableau),
        )
        record!(obs, :step_start, :entry, :L70, 1, :step_start, ctx.t, data)
    end

    function (o::EventObserver)(event::CTS.StageBaseline)
        obs = o.observer
        ctx = event.ctx
        obs.stage = stage_of(ctx)
        data = Dict{Symbol, Any}(:temp => copy(event.temp), :dtγ => ctx.dtγ)
        record!(obs, :baseline, :exit, :baseline, 0, :implicit_setup, ctx.t, data)
    end

    function (o::EventObserver)(event::CTS.EffectiveStorage)
        obs = o.observer
        ctx = event.ctx
        obs.stage = stage_of(ctx)
        data = Dict{Symbol, Any}(
            :T_imp_read => copy(event.T_imp_i),
            :T_imp_index => event.i,
            :U => copy(event.U),
            :temp => copy(event.temp),
            :dtγ => event.dtγ,
        )
        site = :effective_storage
        record!(obs, site, :exit, site, 12, :effective_storage, ctx.t, data)
    end

    function (o::EventObserver)(event::CTS.AssemblyInputs)
        obs = o.observer
        ctx = event.ctx
        obs.stage = nothing
        tab = event.tableau
        data = Dict{Symbol, Any}(
            :u => copy(event.u),
            :dt => event.dt,
            :T_exp_all => copy_columns(event.T_exp, tab.a_exp, tab.b_exp),
            :T_lim_all => copy_columns(event.T_lim, tab.a_exp, tab.b_exp),
            :T_imp_all => copy_columns(event.T_imp, tab.a_imp, tab.b_imp),
            :b_exp => snapshot(event.b_exp),
            :b_imp => snapshot(event.b_imp),
        )
        site = :assembly_inputs
        record!(obs, site, :entry, site, 13, :final_assembly, ctx.t, data)
    end

    function (o::EventObserver)(event::CTS.Operation)
        obs = o.observer
        ctx = event.ctx
        obs.stage = stage_of(ctx)
        data = state(event.state)
        isnothing(event.extra) || (data[:extra] = copy(event.extra))
        data[:dtγ] = ctx.dtγ
        data[:t_exp] = ctx.t_exp
        data[:t_imp] = ctx.t_imp
        hook, point = if haskey(SITE_MAP, event.site)
            mapped = SITE_MAP[event.site]
            (mapped.hook, event.phase == :entry ? mapped.points[1] : mapped.points[2])
        else
            (:operation, 0)
        end
        record!(obs, hook, event.phase, event.site, point, event.role, ctx.t, data)
    end

    # Model-hook wrappers of the event path; the step and stage come from the events.
    function event_wrap_T_exp_T_lim(f!, obs)
        isnothing(f!) && return nothing
        return function (T_exp_i, T_lim_i, U, p, t)
            site, point = if !obs.in_step
                (:outside_step, 0)
            else
                (:L273, obs.stage == 1 ? 2 : 11)
            end
            record!(obs, :T_exp_T_lim!, :entry, site, point, :stage_tendency, t, state(U))
            result = f!(T_exp_i, T_lim_i, U, p, t)
            data = state(U)
            data[:T_exp] = copy(T_exp_i)
            data[:T_lim] = copy(T_lim_i)
            record!(obs, :T_exp_T_lim!, :exit, site, point, :stage_tendency, t, data)
            return result
        end
    end

    function event_wrap_T_imp(f!, obs)
        return function (T_imp_i, U, p, t)
            site = obs.in_step && !isnothing(obs.stage) ? :L311 : :unexpected
            role = :provisional_evaluation
            record!(obs, :T_imp!, :entry, site, 7, role, t, state(U))
            result = f!(T_imp_i, U, p, t)
            data = state(U)
            data[:output] = copy(T_imp_i)
            record!(obs, :T_imp!, :exit, site, 7, role, t, data)
            return result
        end
    end

    function event_wrap_T_post_imp(f!, obs)
        isnothing(f!) && return nothing
        return function (dU, U, p, t)
            role = :correction_scratch
            record!(obs, :T_post_imp!, :entry, :L236, 9, role, t, state(U))
            result = f!(dU, U, p, t)
            data = state(U)
            data[:output] = copy(dU)
            record!(obs, :T_post_imp!, :exit, :L236, 9, role, t, data)
            return result
        end
    end

    """
        wrap_model_hooks(f::CTS.ClimaODEFunction, obs::Observer)

    Return a new `ClimaODEFunction` whose model hooks (`T_exp_T_lim!`, `T_imp!`, `Wfact`,
    `T_post_imp!`) record entry and exit events in `obs`; the solver-called hooks and the
    update-signal policies are passed through unchanged (local result, prototype; B001 T9).
    """
    function wrap_model_hooks(f::CTS.ClimaODEFunction, obs::Observer)
        fields = fieldnames(typeof(f))
        Set(fields) == Set(CLIMA_ODE_FUNCTION_FIELDS) || error(
            "ClimaODEFunction fields $(fields) differ from the pinned 1.0.1 set $(CLIMA_ODE_FUNCTION_FIELDS)",
        )
        T_imp! = if isnothing(f.T_imp!)
            nothing
        else
            CTS.ODEFunction(
                event_wrap_T_imp(f.T_imp!.f, obs);
                jac_prototype = f.T_imp!.jac_prototype,
                Wfact = wrap_Wfact(f.T_imp!.Wfact, obs),
                tgrad = f.T_imp!.tgrad,
            )
        end
        return CTS.ClimaODEFunction(;
            T_exp_T_lim! = event_wrap_T_exp_T_lim(f.T_exp_T_lim!, obs),
            T_imp!,
            T_post_imp! = event_wrap_T_post_imp(f.T_post_imp!, obs),
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

    """
        instrumented_integrator_events(sim, obs::EventObserver; callbacks)

    Like `instrumented_integrator`, but the solver internals come from `CTS.init(...;
    solver_observer = obs)` and only the model hooks are wrapped (`wrap_model_hooks`). The
    observer's step-end callback is appended after `callbacks`. Local result, prototype
    (B001 T9); the stock integrator of `sim` must not be stepped.
    """
    function instrumented_integrator_events(
        sim,
        obs::EventObserver;
        callbacks = sim.integrator.callback.discrete_callbacks[1:(end - 1)],
    )
        stock = sim.integrator
        prob = stock.sol.prob
        inner = obs.observer
        step_end_callback = CTS.DiscreteCallback(
            (u, t, integrator) -> true,
            integrator -> record_step_end!(inner, integrator),
        )
        return CTS.init(
            CTS.ODEProblem(wrap_model_hooks(prob.f, inner), prob.u0, prob.tspan, prob.p),
            stock.alg;
            dt = stock._dt,
            saveat = stock._saveat,
            callback = CTS.CallbackSet(callbacks..., step_end_callback),
            solver_observer = obs,
        )
    end

end

end
