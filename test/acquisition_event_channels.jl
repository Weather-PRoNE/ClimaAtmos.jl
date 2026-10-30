using Test
import LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS
import ClimaCore: CommonSpaces, Fields, Geometry, Spaces

# Standalone event-channel checks of B001 acceptance claims 2–4 on the ClimaAtmos side
# (B001 T9 part 2, C037); not part of runtests.jl. Needs a ClimaTimeSteppers with labelled
# solver events. Expected values are specified alongside the fixtures below, with
# derivations for selected checks and exact identities for others.
println("CTS from ", pathof(CTS))

include("acquisition_helpers.jl")
include("solver_event_consumer.jl")
report(name, value) = println(rpad(name, 60), " = ", value)

# Fixture-local wrappers of the implicit tendency and the correction, which CTS does not
# report per evaluation. Each calls the original hook once and records copies under the
# recorder's current stage.
function recorded_T_imp(f!, rec)
    return function (T, U, p, t)
        role = :provisional_evaluation
        record_hook!(rec, :T_imp!, :entry, role, (; state = copy(U)))
        result = f!(T, U, p, t)
        record_hook!(rec, :T_imp!, :exit, role, (; state = copy(U), output = copy(T)))
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

function wrap_hooks(f, rec)
    T_imp! = if isnothing(f.T_imp!)
        nothing
    else
        CTS.ODEFunction(
            recorded_T_imp(f.T_imp!.f, rec);
            jac_prototype = f.T_imp!.jac_prototype,
            Wfact = f.T_imp!.Wfact,
            tgrad = f.T_imp!.tgrad,
        )
    end
    return CTS.ClimaODEFunction(;
        f.T_exp_T_lim!,
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

# Runs one ARS343 step of `f` from `u0` through the hook wrappers and the event recorder.
function observed_step(f, u0, dt, p)
    recorder = EventRecorder()
    prob = CTS.ODEProblem(wrap_hooks(f, recorder), u0, (zero(dt), dt), p)
    alg = CTS.IMEXAlgorithm(CTS.ARS343(), CTS.NewtonsMethod(; max_iters = 1))
    integrator = CTS.init(prob, alg; dt, solver_observer = recorder)
    CTS.step!(integrator)
    return integrator, recorder.records
end

# Scalar Jacobian `W = dtγ ∂T/∂u − 1` of a one-element state, so Newton solves `W Δ = R`.
struct ScalarW{FT}
    w::Base.RefValue{FT}
end
Base.similar(W::ScalarW) = ScalarW(Ref(W.w[]))
Base.zero(W::ScalarW) = ScalarW(Ref(zero(W.w[])))
LinearAlgebra.ldiv!(x, W::ScalarW, b) = (x .= b ./ W.w[]; x)

#=
Forward-error bound used for the non-exact scalar checks (claims 3 and 4). All targets have
magnitude ≤ 9 and are computed in the test precision FT by at most 16 rounded operations:
dt and dtγ formation (≤ 3), the residual and Jacobian (≤ 4), the Newton update (2), the
correction update (2), the effective storage (U − temp)/dtγ (2) and a ≤ 4-term weighted
assembly. Each rounding adds ≤ eps(FT)/2 relative to an operand of magnitude ≤ 9, so the
absolute error before the storage division is ≤ 16 · 9 · eps/2 = 72 eps; dividing by
dtγ ≥ 1/4 amplifies it by ≤ 4, giving ≤ 288 eps. We use 512 eps(FT) (Float32: 6.1e-5,
Float64: 1.1e-13). Every fault magnitude below (≥ 1/144 ≈ 6.9e-3) exceeds it.
=#
bound(::Type{FT}) where {FT} = 512 * eps(FT)

limiter_events(events, site) = [e for e in events if e.site == site]

function claim2(::Type{FT}; fault = false) where {FT}
    cspace = CommonSpaces.ColumnSpace(
        FT;
        z_elem = 3,
        z_min = 0,
        z_max = 3,
        staggering = CommonSpaces.CellCenter(),
    )
    fspace = Spaces.face_space(cspace)
    c = fill((; ρ = FT(1.25), ρe_tot = FT(2.5)), cspace)
    f = fill((; u₃ = Geometry.Covariant3Vector(FT(0.5))), fspace)
    Y = Fields.FieldVector(; c, f)
    p = (;
        numerics = (;
            sem_quasimonotone_limiter = nothing,
            vertical_water_borrowing_limiter = nothing,
            vertical_water_borrowing_species = nothing,
        ),
    )
    refs = []
    lim! = function (Y, p, t, ref_Y)
        push!(refs, copy(ref_Y))
        CA.limiters_func!(Y, p, t, ref_Y)
        # Fault: one density slot changes by exactly 1.
        fault && (parent(Y.c.ρ)[1] += 1)
        return nothing
    end
    T_exp_T_lim! = function (T_exp, T_lim, U, p, t)
        fill!(T_exp, zero(FT))
        fill!(T_lim, zero(FT))
        return nothing
    end
    odef = CTS.ClimaODEFunction(; T_exp_T_lim!, lim!)
    _, events = observed_step(odef, Y, FT(1), p)
    return events, refs
end

function check_claim2(::Type{FT}) where {FT}
    events, refs = claim2(FT)
    stage = limiter_events(events, :stage_lim)
    final = limiter_events(events, :final_lim)
    pairs = [
        (only(e for e in es if e.stage == s && e.phase == :entry),
            only(e for e in es if e.stage == s && e.phase == :exit)) for
        (es, s) in ((stage, 2), (stage, 3), (stage, 4), (final, 0))
    ]
    report(
        "C037 claim 2 $FT limiter calls (stage, final, hook)",
        (length(stage), length(final), length(refs)),
    )
    @test length(stage) == 6 && length(final) == 2
    @test sort([e.stage for e in stage if e.phase == :entry]) == [2, 3, 4]
    @test length(refs) == 4
    for (k, (entry, exit)) in enumerate(pairs)
        @test exact_equal(entry.data[:state], exit.data[:state])
        @test exact_equal(entry.data[:extra], refs[k])
    end
    events_f, _ = claim2(FT; fault = true)
    stage_f = limiter_events(events_f, :stage_lim)
    entry = only(e for e in stage_f if e.stage == 2 && e.phase == :entry)
    exit = only(e for e in stage_f if e.stage == 2 && e.phase == :exit)
    report(
        "C037 claim 2 $FT fault max difference",
        maxdiff(entry.data[:state], exit.data[:state]),
    )
    @test !exact_equal(entry.data[:state], exit.data[:state])
    @test maxdiff(entry.data[:state], exit.data[:state]) == 1
end

# T0 Section 3: u0 = 0, dt = 1, explicit rate 1, implicit rate 2, correction rate 3.
function claim3(::Type{FT}; correction) where {FT}
    T_exp_T_lim! = function (T_exp, T_lim, U, p, t)
        T_exp .= 1
        T_lim .= 0
        return nothing
    end
    T_imp = (T, U, p, t) -> (T .= 2; nothing)
    Wfact = (W, U, p, dtγ, t) -> (W.w[] = -1; nothing)
    T_post_imp! = correction ? ((dU, U, p, t) -> (dU .= 3; nothing)) : nothing
    odef = CTS.ClimaODEFunction(;
        T_exp_T_lim!,
        T_imp! = CTS.ODEFunction(T_imp; jac_prototype = ScalarW(Ref(zero(FT))), Wfact),
        T_post_imp!,
    )
    integrator, events = observed_step(odef, FT[0], FT(1), nothing)
    return integrator.u, events
end

# Top-level reconstruction from the assembly inputs; the effective implicit storage already
# contains the correction, so its subordinate children are not added again.
function reconstruct(asm)
    u, dt, b_exp, b_imp = (asm.data[k] for k in (:u, :dt, :b_exp, :b_imp))
    T_exp_all, T_imp_all = asm.data[:T_exp], asm.data[:T_imp]
    return u[1] +
           dt * (
        sum(b_exp[j] * T[1] for (j, T) in T_exp_all) +
        sum(b_imp[j] * T[1] for (j, T) in T_imp_all)
    )
end

function check_claim3(::Type{FT}) where {FT}
    tol = bound(FT)
    for (correction, rate, endpoint) in ((false, 2, 3), (true, 5, 6))
        u, events = claim3(FT; correction)
        stored = [e for e in events if e.kind == :EffectiveStorage]
        @test sort([e.stage for e in stored]) == [2, 3, 4]
        @test all(abs(e.data[:T_imp_i][1] - rate) <= tol for e in stored)
        @test abs(u[1] - endpoint) <= tol
        asm = only(e for e in events if e.kind == :AssemblyInputs)
        @test abs(reconstruct(asm) - endpoint) <= tol
        # Hook-wrapper records of the correction; the solver's applied-correction operation
        # records carry the site :post_imp_correction.
        children = [e for e in events if e.kind == :hook && e.site == :T_post_imp!]
        ops = [e for e in events if e.site == :post_imp_correction]
        report(
            "C037 claim 3 $FT correction=$correction (endpoint, children, ops)",
            (u[1], length(children), length(ops)),
        )
        if !correction
            # Off means absent correction calls, not zero-valued ones.
            @test isempty(children) && isempty(ops)
            continue
        end
        outputs = [e for e in children if e.phase == :exit]
        @test sort([e.stage for e in outputs]) == [2, 3, 4]
        @test all(e.data[:output] == FT[3] for e in outputs)
        @test all(e.role == :correction_scratch for e in children)
        # Signed stage children 3b₁, 3b₂, 3γ summing to the correction total 3, with the
        # closed-form ARS343 weights b₁ = −3γ²/2 + 4γ − 1/4 > 0, b₂ = 3γ²/2 − 5γ + 5/4 < 0
        # (γ = dtγ at dt = 1).
        b_imp = asm.data[:b_imp]
        outputs = sort(outputs; by = e -> e.stage)
        weighted = [asm.data[:dt] * b_imp[e.stage] * e.data[:output][1] for e in outputs]
        γ = stored[1].data[:dtγ]
        expected = 3 .* (-3γ^2 / 2 + 4γ - 1 // 4, 3γ^2 / 2 - 5γ + 5 // 4, γ)
        report("C037 claim 3 $FT children (acquired, closed form)", (weighted, expected))
        @test sign.(weighted) == [1, -1, 1]
        @test all(abs.(weighted .- expected) .<= tol)
        @test abs(sum(weighted) - 3) <= tol
        # Fault: the children added alongside the effective total give 9 vs the acquired 6.
        faulty = reconstruct(asm) + sum(weighted)
        report("C037 claim 3 $FT fault (reconstruction, residual)", (faulty, u[1] - faulty))
        @test abs(faulty - 9) <= tol
        @test abs((u[1] - faulty) - (-3)) <= tol
        @test abs(u[1] - faulty) > tol
    end
end

# T0 Section 4: f_exp = 0, f_imp(u) = −u², u0 = 1, dt = 1/(4γ), one undamped Newton step.
function claim4(::Type{FT}) where {FT}
    T_exp_T_lim! = function (T_exp, T_lim, U, p, t)
        T_exp .= 0
        T_lim .= 0
        return nothing
    end
    T_imp = (T, U, p, t) -> (T .= .-U .^ 2; nothing)
    Wfact = (W, U, p, dtγ, t) -> (W.w[] = dtγ * (-2 * U[1]) - 1; nothing)
    odef = CTS.ClimaODEFunction(;
        T_exp_T_lim!,
        T_imp! = CTS.ODEFunction(T_imp; jac_prototype = ScalarW(Ref(zero(FT))), Wfact),
    )
    # γ of ARS343 is read from a dt = 1 pilot step's stage-2 baseline dtγ.
    _, pilot = observed_step(odef, FT[1], FT(1), nothing)
    γ = only(e for e in pilot if e.kind == :StageBaseline && e.stage == 2).ctx.dtγ
    _, events = observed_step(odef, FT[1], 1 / (4 * γ), nothing)
    return events
end

function check_claim4(::Type{FT}) where {FT}
    tol = bound(FT)
    events = claim4(FT)
    evals = [e for e in events if e.site == :T_imp! && e.stage == 2 && e.phase == :exit]
    stored = only(e for e in events if e.kind == :EffectiveStorage && e.stage == 2)
    h = stored.data[:dtγ]
    B = stored.data[:temp][1]
    U = stored.data[:U][1]
    T_eff = stored.data[:T_imp_i][1]
    # The one recorded physical evaluation is at the initial guess, state 1, output −1.
    @test length(evals) == 1
    recorded = only(evals)
    @test recorded.role == :provisional_evaluation
    @test recorded.data[:state] == FT[1]
    @test recorded.data[:output] == FT[-1]
    @test abs(h - 1 // 4) <= tol
    @test B == 1
    @test abs(U - 5 // 6) <= tol
    @test abs(T_eff - (-2 // 3)) <= tol
    # Diagnostic probe, not a solver evaluation: the manufactured f_imp at the retained U.
    probe = -U^2
    D = U - B - h * probe
    report("C037 claim 4 $FT (U, T_eff, probe f_imp(U), D)", (U, T_eff, probe, D))
    @test abs(probe - (-25 // 36)) <= tol
    @test abs(D - 1 // 144) <= tol
    @test abs((T_eff - probe) - 1 // 36) <= tol
    @test abs((T_eff - probe) - D / h) <= tol
    # Faults: each substitution misses its oracle by more than the bound.
    faults = (
        physical_at_final = abs(probe - (-2 // 3)),
        zero_defect = abs(0 - D),
        recorded_eval = abs(recorded.data[:output][1] - (-2 // 3)),
    )
    report("C037 claim 4 $FT faults", faults)
    @test abs(faults.physical_at_final - 1 // 36) <= tol
    @test abs(faults.zero_defect - 1 // 144) <= tol
    @test abs(faults.recorded_eval - 1 // 3) <= tol
    @test all(>(tol), faults)
end

@testset "C037 event channels" begin
    for FT in (Float32, Float64)
        @testset "claim 2 inactive limiter $FT" check_claim2(FT)
        @testset "claim 3 correction on/off $FT" check_claim3(FT)
        @testset "claim 4 finite Newton $FT" check_claim4(FT)
    end
end

# Wiring smoke: `energy_q_tot_upwinding = none` leaves no correction hook.
@testset "C037 claim 3 wiring smoke" begin
    hook(upwinding) =
        CA.get_simulation(
            CA.AtmosConfig(
                Dict(
                    "config" => "column",
                    "z_elem" => 4,
                    "dt" => "1secs",
                    "t_end" => "1secs",
                    "energy_q_tot_upwinding" => upwinding,
                    "log_progress" => false,
                    "enable_diagnostics" => false,
                    "output_default_diagnostics" => false,
                    "output_dir" => mktempdir(),
                );
                job_id = "c037_wiring_$upwinding",
            ),
        ).integrator.sol.prob.f.T_post_imp!
    t0 = time()
    @test hook("none") === nothing
    @test hook("first_order") !== nothing
    report("C037 wiring smoke seconds", round(time() - t0; digits = 1))
end
