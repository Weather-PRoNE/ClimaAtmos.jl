using Test
import LinearAlgebra
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS
import ClimaCore: CommonSpaces, Fields, Geometry, Spaces
import ClimaCore.Utilities: half

# Standalone event-channel checks of B001 acceptance claim 5 on the ClimaAtmos side (B001 T9
# part 2, C037 T3); not part of runtests.jl. Needs a ClimaTimeSteppers with labelled solver
# events. Expected values are hand-derived from the fixtures (derivations at each fixture
# below) or exact identities.
const AP = CA.AcquisitionPrototype
println("CTS from ", pathof(CTS))

include("acquisition_helpers.jl")
report(name, value) = println(rpad(name, 60), " = ", value)

# `prepare!` acts on the state after `CTS.init`, whose own `cache!` call would otherwise
# already apply the boundary setters to the initial state.
function observed_step(f, u0, dt, p; prepare! = identity)
    eobs = AP.EventObserver()
    prob = CTS.ODEProblem(AP.wrap_model_hooks(f, eobs.observer), u0, (zero(dt), dt), p)
    alg = CTS.IMEXAlgorithm(CTS.ARS343(), CTS.NewtonsMethod(; max_iters = 1))
    integrator = CTS.init(prob, alg; dt, solver_observer = eobs)
    prepare!(integrator.u)
    CTS.step!(integrator)
    return integrator, eobs.observer.events
end

struct ScalarW{FT}
    w::Base.RefValue{FT}
end
Base.similar(W::ScalarW) = ScalarW(Ref(W.w[]))
Base.zero(W::ScalarW) = ScalarW(Ref(zero(W.w[])))
LinearAlgebra.ldiv!(x, W::ScalarW, b) = (x .= b ./ W.w[]; x)

#=
Bound for the DSS checks (the only non-exact values here). `weighted_dss!` forms each nodal
weight w = JW / ΣJW (≤ 2 roundings, relative), multiplies it by a value of magnitude ≤ 3
(1 rounding) and sums two terms (1 rounding); the face covariant vectors are additionally
transformed to and from the Cartesian basis (≤ 4 roundings). With ≤ 8 roundings of relative
size eps(FT)/2 on magnitudes ≤ 3 the absolute error is ≤ 8 · 3 · eps/2 = 12 eps; we use
64 eps(FT) (Float32: 7.6e-6, Float64: 1.4e-14). Both DSS fault magnitudes (1) exceed it.
The boundary, no-DSS, count and role checks are exact.
=#
bound(::Type{FT}) where {FT} = 64 * eps(FT)

zero_rates! = function (T_exp, T_lim, U, p, t)
    fill!(T_exp, 0)
    fill!(T_lim, 0)
    return nothing
end

# A step whose tendencies vanish, so the final assembly input equals the step-start state.
function map_step(
    Y,
    p;
    dss!,
    cache!,
    cache_imp! = (Y, p, t) -> nothing,
    implicit = false,
    prepare! = identity,
)
    FT = eltype(parent(Y.c))
    T_imp! = if implicit
        CTS.ODEFunction(
            (T, U, p, t) -> (fill!(T, 0); nothing);
            jac_prototype = ScalarW(Ref(zero(FT))),
            Wfact = (W, U, p, dtγ, t) -> (W.w[] = -1; nothing),
        )
    else
        nothing
    end
    odef = CTS.ClimaODEFunction(;
        T_exp_T_lim! = zero_rates!,
        T_imp!,
        dss!,
        constrain_state! = (Y, p, t) -> nothing,
        cache!,
        cache_imp!,
        update_cache = CTS.UpdateEvery(CTS.EndOfStep),
        update_constrain_state = CTS.UpdateEvery(CTS.EndOfStep),
    )
    return observed_step(odef, Y, FT(1), p; prepare!)
end

ghost_buffers(Y) =
    (; c = Spaces.create_dss_buffer(Y.c), f = Spaces.create_dss_buffer(Y.f))

const FINAL_TEMPLATE = [
    (:final_dss, :entry),
    (:final_dss, :exit),
    (:final_constrain, :entry),
    (:final_constrain, :exit),
    (:final_cache, :entry),
    (:final_cache, :exit),
]
const STAGE_CACHE_SITES = (
    :stage_cache_post,
    :stage_cache_no_implicit,
    :cache_imp_pre_newton,
    :cache_imp_post_newton,
    :stage_cache_fallback,
)

# Generic test consumer: the top-level final maps are exactly DSS → constraint → cache, each
# once, and every stage cache write is a subordinate child inside its stage.
function accounting_accepted(events)
    final = [(e.site, e.phase) for e in events if e.role == :final_map]
    children = [e for e in events if e.site in STAGE_CACHE_SITES]
    return final == FINAL_TEMPLATE && all(
        e.role == :subordinate_stage_write && e.stage isa Int && 1 <= e.stage <= 4 for
        e in children
    )
end

final_pair(events, site) = (
    only(e for e in events if e.site == site && e.phase == :entry).data[:state],
    only(e for e in events if e.site == site && e.phase == :exit).data[:state],
)

promote(e) =
    AP.Event(e.step, nothing, e.point, e.hook, e.phase, e.site, :final_map, e.t, e.data)

# Covariant vertical component on one face level, through `Fields.level`.
u₃_level(Y, l) = only(parent(Fields.level(Y.f.u₃.components.data.:1, l)))

# T0 Section 5: flat Cartesian column, Δz = 1 m, two cells; boundary u₃ = 9 injected, interior
# sentinel 5, uₕ³ = 0. The boundary setters give u₃ = −uₕ³/g³³ = −0.0 at both boundary faces.
function boundary_run(::Type{FT}; skip_top = false, stage_writes = false) where {FT}
    cspace = CommonSpaces.ColumnSpace(
        FT;
        z_elem = 2,
        z_min = 0,
        z_max = 2,
        staggering = CommonSpaces.CellCenter(),
    )
    fspace = Spaces.face_space(cspace)
    c = fill((; ρ = FT(1.25), ρe_tot = FT(2.5)), cspace)
    f = fill((; u₃ = Geometry.Covariant3Vector(FT(5))), fspace)
    Y = Fields.FieldVector(; c, f)
    inject! = function (Y)
        for l in (half, 2 + half)
            Fields.level(Y.f.u₃.components.data.:1, l) .= 9
        end
        return Y
    end
    ᶠuₕ³ = fill(Geometry.Contravariant3Vector(FT(0)), fspace)
    p = (; ᶠuₕ³, ghost_buffer = ghost_buffers(Y), scratch = Ref(0))
    cache! = function (Y, p, t)
        CA.set_velocity_at_surface!(Y, p.ᶠuₕ³, nothing)
        skip_top || CA.set_velocity_at_top!(Y, p.ᶠuₕ³, nothing)
        return nothing
    end
    # Step-only policies keep `cache!` out of the stages; the implicit part still calls
    # `cache_imp!` there (T3-research Finding 5). As a no-op it leaves the effective tendencies
    # zero, so `Y_asm` equals `u`; with `stage_writes` it is the production boundary setter.
    cache_imp! = stage_writes ? cache! : (Y, p, t) -> nothing
    return map_step(
        Y,
        p;
        dss! = CA.dss!,
        cache!,
        cache_imp!,
        implicit = true,
        prepare! = inject!,
    )
end

function check_boundary(::Type{FT}) where {FT}
    integrator, events = boundary_run(FT)
    Y_next = integrator.u
    entry, exit = final_pair(events, :final_cache)
    top = 2 + half
    values = (
        u₃_level(entry, half),
        u₃_level(entry, top),
        u₃_level(exit, half),
        u₃_level(exit, top),
    )
    report("C037 claim 5 $FT boundary u₃ (entry sfc, top; exit sfc, top)", values)
    @test u₃_level(entry, half) == 9 && u₃_level(entry, top) == 9
    # Limit: with uₕ³ = 0 the oracle −uₕ³/g³³ = −0.0 does not tell the formula from a hard
    # zero; the sloped-terrain case is not tested here.
    @test u₃_level(exit, half) == FT(-0.0) && u₃_level(exit, top) == FT(-0.0)
    @test u₃_level(exit, half) - u₃_level(entry, half) == -9
    @test u₃_level(exit, top) - u₃_level(entry, top) == -9
    @test u₃_level(exit, 1 + half) == u₃_level(entry, 1 + half) == 5
    @test parent(exit.c) == parent(entry.c)
    @test exact_equal(Y_next, exit)

    # Ordered final maps, each once; stage cache writes are subordinate and precede the
    # final assembly inputs.
    @test accounting_accepted(events)
    children = [(k, e) for (k, e) in enumerate(events) if e.site in STAGE_CACHE_SITES]
    report(
        "C037 claim 5 $FT stage cache events (count, sites)",
        (length(children), unique(e.site for (_, e) in children)),
    )
    @test !isempty(children)
    @test all(e.role == :subordinate_stage_write for (_, e) in children)
    ka = findfirst(x -> x.site == :assembly_inputs, events)
    @test all(k < ka for (k, _) in children)
    # The step-start state equals the assembly input, since all rates vanish.
    Y_asm, dss_exit = final_pair(events, :final_dss)
    @test exact_equal(Y_asm, only(e for e in events if e.hook == :step_start).data[:u])

    # Top-level reconstruction from the final-map increments, without the stage children.
    increments = [
        x .- y for (y, x) in
        (final_pair(events, s) for s in (:final_dss, :final_constrain, :final_cache))
    ]
    recon = Y_asm .+ increments[1] .+ increments[2] .+ increments[3]
    @test exact_equal(recon, Y_next)
    # Fault: omit the final-cache increment → face residual 9.
    omitted = Y_asm .+ increments[1] .+ increments[2]
    report(
        "C037 claim 5 $FT fault omit cache increment (face residual)",
        maxdiff(omitted.f, Y_next.f),
    )
    @test maxdiff(omitted.f, Y_next.f) == 9
    # Fault: a stage child promoted to a second top-level contribution is rejected.
    promoted = vcat(events, [promote(e) for (_, e) in children[1:2]])
    @test !accounting_accepted(promoted)
    # Fault: skip the top setter → boundary error 9.
    integrator_f, events_f = boundary_run(FT; skip_top = true)
    _, exit_f = final_pair(events_f, :final_cache)
    err = abs(u₃_level(exit_f, top) - 0)
    report("C037 claim 5 $FT fault skip top setter (error)", err)
    @test err == 9
end

#=
Accounting-consistency check, not an independent numerical oracle: with the boundary setters
as `cache_imp!` the stage writes are nonzero and enter the assembly through the effective
implicit tendencies. The reconstruction u + dt Σ (b_exp T_exp + b_exp T_lim + b_imp T_imp)
from the observed assembly inputs and the solver's own Y_asm are two evaluations of the same
≤ 13-term sum (4 + 4 explicit and limiter, 4 implicit, plus u); each has absolute error
≤ 14 · eps/2 · S with S = |u| + dt Σ |b T|, so their difference is ≤ 14 eps S; we use 32 eps S.
=#
function check_stage_writes(::Type{FT}) where {FT}
    integrator, events = boundary_run(FT; stage_writes = true)
    a = only(e for e in events if e.site == :assembly_inputs).data
    weighted(b, T) = [(b[j], T[j]) for j in keys(T)]
    all_terms = vcat(
        weighted(a[:b_exp], a[:T_exp_all]),
        weighted(a[:b_exp], a[:T_lim_all]),
        weighted(a[:b_imp], a[:T_imp_all]),
    )
    recon = copy(a[:u])
    scale = abs.(a[:u])
    for (b, T) in all_terms
        recon .+= a[:dt] .* b .* T
        scale .+= abs.(a[:dt] .* b .* T)
    end
    S = maximum(maximum(parent(x)) for x in components(scale))
    tol = 32 * eps(FT) * S
    Y_asm, _ = final_pair(events, :final_dss)
    # The stage writes are contained in the effective storage of the implicit stages.
    eff = [e.data[:T_imp_read] for e in events if e.site == :effective_storage]
    report(
        "C037 claim 5 $FT stage writes: max |effective T_imp| per stage",
        [maximum(abs, parent(x.f)) for x in eff],
    )
    @test any(maximum(abs, parent(x.f)) > tol for x in eff)
    report(
        "C037 claim 5 $FT stage writes: (|recon − Y_asm|, bound)",
        (maxdiff(recon, Y_asm), tol),
    )
    @test maxdiff(recon, Y_asm) <= tol
    # Fault: adding the subordinate children again counts each stage write twice.
    children = [e for e in events if e.site in STAGE_CACHE_SITES && e.phase == :exit]
    entries = [e for e in events if e.site in STAGE_CACHE_SITES && e.phase == :entry]
    double = copy(recon)
    for (x, y) in zip(children, entries)
        double .+= x.data[:state] .- y.data[:state]
    end
    report(
        "C037 claim 5 $FT fault re-add stage children (deviation, children)",
        (maxdiff(double, Y_asm), length(children)),
    )
    @test maxdiff(double, Y_asm) > tol

    #=
    Weighted re-add of the stage children, against a hand derivation from the ARS343 tableau
    (γ = g, a_imp[3, 2] = d = (1 − g)/2, b_imp = (0, b₁, b₂, g)) and this fixture (zero
    rates, boundary u = 9, dt = 1). Stage i ∈ 2:4 starts from the boundary value Bᵢ = Ûᵢ; its
    cache writes record Bᵢ → 0, Bᵢ (Newton restores Ûᵢ under a zero right-hand side) and
    → 0 again, so Σₖ Δu₃ = −2Bᵢ, while its effective tendency is Tᵢ = −Bᵢ/(dt g). Hence
    B₂ = 9, B₃ = 9 − 9d/g, B₄ = 9 − (b₁B₂ + b₂B₃)/g, so b₁B₂ + b₂B₃ + gB₄ = 9g and
    Σᵢ (dt bᵢ/(dt g)) Σₖ Δu₃ = −(2/g)(b₁B₂ + b₂B₃ + gB₄) = −18 at each boundary face.
    Bound: each Bᵢ comes from ≤ 8 roundings of the solver's stage arithmetic, each weight
    from ≤ 2 and each weighted term from ≤ 2 more; with ≤ 12 roundings of relative size
    eps/2, propagated through weights |bᵢ/g| ≤ 3, the error is ≤ 18 eps S with
    S = Σᵢ |wᵢ| Σₖ |Δu₃|; we use 64 eps(FT) S. The unweighted sum −2(B₂ + B₃ + B₄) ≈ −1.83
    and the sum without 1/dtγ, −2 · 9g ≈ −7.85, both miss −18 by far more.
    =#
    g = 0.4358665215084590
    b_imp = FT.((0, -3 / 2 * g^2 + 4g - 1 / 4, 3 / 2 * g^2 - 5g + 5 / 4, g))
    dt = a[:dt]
    dtγ = dt * FT(g)
    @test maximum(abs(a[:b_imp][j] - b_imp[j]) for j in 1:4) <= 4 * eps(FT)
    for l in (half, 2 + half)
        weighted_sum = unweighted = no_dtγ = scale_l = zero(FT)
        outside = zero(FT)
        for (x, y) in zip(children, entries)
            @test x.site == y.site && x.stage == y.stage
            Δ = u₃_level(x.data[:state], l) - u₃_level(y.data[:state], l)
            unweighted += Δ
            if x.stage in 2:4
                w = dt * b_imp[x.stage] / dtγ
                weighted_sum += w * Δ
                no_dtγ += dt * b_imp[x.stage] * Δ
                scale_l += abs(w) * abs(Δ)
            else
                outside += abs(Δ)
            end
        end
        tol_w = 64 * eps(FT) * scale_l
        report(
            "C039 claim 5(b) $FT face $l weighted re-add (sum, bound)",
            (weighted_sum, tol_w),
        )
        report(
            "C039 claim 5(b) $FT face $l faults (unweighted, no 1/dtγ)",
            (unweighted, no_dtγ),
        )
        @test outside == 0
        @test abs(weighted_sum - (-18)) <= tol_w
        # Faults: the unweighted sum and the sum without 1/dtγ.
        @test abs(unweighted - (-18)) > tol_w
        @test abs(no_dtγ - (-18)) > tol_w
    end
end

# Nodal value at GLL point i of horizontal element h, level v.
# The parent arrays are (v, i, j, f, h), observed as 1×2×1×1×2 for this space.
node(field, i, h; v = 1) = parent(field)[v, i, 1, 1, h]
set_node!(field, x, i, h; v = 1) = (parent(field)[v, i, 1, 1, h] = x)

# T0 Section 5: two congruent GLL elements (2 points each) on the non-periodic x ∈ (0, 2), one
# vertical cell. Element 1 holds 0, element 2 holds 2, so the shared interface is (0, 2).
function dss_run(::Type{FT}; skip = false) where {FT}
    cspace = CommonSpaces.SliceXZSpace(
        FT;
        z_min = 0,
        z_max = 1,
        z_elem = 1,
        x_min = 0,
        x_max = 2,
        x_elem = 2,
        periodic_x = false,
        n_quad_points = 2,
        staggering = CommonSpaces.CellCenter(),
    )
    fspace = Spaces.face_space(cspace)
    c = fill((; ρ = FT(0)), cspace)
    f = fill((; u₃ = Geometry.Covariant3Vector(FT(3))), fspace)
    Y = Fields.FieldVector(; c, f)
    foreach(i -> set_node!(Y.c.ρ, FT(2), i, 2), 1:2)
    p = (; ghost_buffer = ghost_buffers(Y))
    dss! = skip ? (Y, p, t) -> nothing : CA.dss!
    _, events = map_step(Y, p; dss!, cache! = (Y, p, t) -> nothing)
    return cspace, events
end

function check_dss(::Type{FT}) where {FT}
    tol = bound(FT)
    cspace, events = dss_run(FT)
    # Equal native DSS weights on both sides of the interface, established before the check.
    # The weight layout has singleton level/j/field dimensions here, so its parent in memory
    # order is (i, h): entries 2 and 3 are point 2 of element 1 and point 1 of element 2.
    w = parent(Spaces.dss_weights(Spaces.horizontal_space(cspace)))
    w_left, w_right = vec(w)[2], vec(w)[3]
    report(
        "C037 claim 5 $FT interface DSS weights (parent size, left, right)",
        (size(w), w_left, w_right),
    )
    @test length(w) == 4
    @test w_left == w_right
    entry, exit = final_pair(events, :final_dss)
    before = (node(entry.c.ρ, 2, 1), node(entry.c.ρ, 1, 2))
    after = (node(exit.c.ρ, 2, 1), node(exit.c.ρ, 1, 2))
    report("C037 claim 5 $FT DSS interface (before, after)", (before, after))
    @test before == (0, 2)
    @test all(abs.(after .- (1, 1)) .<= tol)
    @test all(abs.((after .- before) .- (1, -1)) .<= tol)
    # The unshared end nodes and the uniform face field stay (within the bound).
    @test abs(node(exit.c.ρ, 1, 1) - 0) <= tol && abs(node(exit.c.ρ, 2, 2) - 2) <= tol
    @test maxdiff(exit.f, entry.f) <= tol
    @test accounting_accepted(events)
    # Duplicated DSS is idempotent in value, so it is rejected by the event count and order.
    k = findfirst(e -> e.site == :final_dss && e.phase == :exit, events)
    duplicated = vcat(events[1:k], events[(k - 1):k], events[(k + 1):end])
    @test !accounting_accepted(duplicated)
    # Fault: skip DSS → interface error 1.
    _, events_f = dss_run(FT; skip = true)
    _, exit_f = final_pair(events_f, :final_dss)
    err = maximum(abs.((node(exit_f.c.ρ, 2, 1), node(exit_f.c.ρ, 1, 2)) .- (1, 1)))
    report("C037 claim 5 $FT fault skip DSS (interface error)", err)
    @test err == 1 && err > tol
end

# A finite-difference column is a no-DSS space: the final DSS leaves the state exactly alone.
function nodss_run(::Type{FT}; fault = false) where {FT}
    cspace = CommonSpaces.ColumnSpace(
        FT;
        z_elem = 2,
        z_min = 0,
        z_max = 2,
        staggering = CommonSpaces.CellCenter(),
    )
    c = fill((; ρ = FT(0)), cspace)
    f = fill((; u₃ = Geometry.Covariant3Vector(FT(3))), Spaces.face_space(cspace))
    Y = Fields.FieldVector(; c, f)
    Fields.level(Y.c.ρ, 2) .= 2
    p = (; ghost_buffer = ghost_buffers(Y))
    dss! = function (Y, p, t)
        CA.dss!(Y, p, t)
        # Fault: one slot changes by exactly 1.
        fault && (Fields.level(Y.c.ρ, 1) .+= 1)
        return nothing
    end
    _, events = map_step(Y, p; dss!, cache! = (Y, p, t) -> nothing)
    return final_pair(events, :final_dss)
end

function check_nodss(::Type{FT}) where {FT}
    entry, exit = nodss_run(FT)
    @test exact_equal(entry, exit)
    entry_f, exit_f = nodss_run(FT; fault = true)
    report("C037 claim 5 $FT fault no-DSS slot (max difference)", maxdiff(entry_f, exit_f))
    @test !exact_equal(entry_f, exit_f)
    @test maxdiff(entry_f, exit_f) == 1
end

@testset "C037 event maps" begin
    for FT in (Float32, Float64)
        @testset "claim 5 boundary u₃ $FT" check_boundary(FT)
        @testset "claim 5 stage-write accounting $FT" check_stage_writes(FT)
        @testset "claim 5 DSS $FT" check_dss(FT)
        @testset "claim 5 no-DSS $FT" check_nodss(FT)
    end
end
