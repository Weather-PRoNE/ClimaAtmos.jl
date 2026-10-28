using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS
import ClimaCore: CommonSpaces, Fields, Geometry, Operators, Spaces
import ClimaCore.Utilities: half

# Standalone event-channel checks of B001 acceptance claim 6 on the ClimaAtmos side (B001 T9
# part 2, C037 T3); not part of runtests.jl. Needs a ClimaTimeSteppers with labelled solver
# events. Expected values are hand-derived from the states below; these are diagnostic
# states, not an atmospheric evolution.
const AP = CA.AcquisitionPrototype
println("CTS from ", pathof(CTS))

report(name, value) = println(rpad(name, 60), " = ", value)

#=
Forward-error bound in the test precision FT. `compute_kinetic` forms the horizontal norm
through the metric (≤ 4 roundings), u₃u³ on faces (≤ 2), the face-to-center interpolation
(≤ 2), the cross term (≤ 4) and the final sum (≤ 2): ≤ 14 roundings of relative size eps/2
on magnitudes ≤ 36, so |δK| ≤ 14 · 36 · eps/2 = 252 eps. The densities multiply by ρ ≤ 3 and
take ≤ 3 more roundings on magnitudes ≤ 330: |δE| ≤ 3 · 252 eps + 3 · 330 eps/2 < 1300 eps.
The column integrals add two terms with unit cell Jacobian (≤ 2 roundings on magnitude ≤ 2).
We use 4096 eps(FT) (Float32: 4.9e-4, Float64: 9.1e-13) everywhere. The fault magnitudes
(4 J m⁻³ and 1 J m⁻³) exceed it.
=#
bound(::Type{FT}) where {FT} = 4096 * eps(FT)

# T0 Section 6 states: (ρ, uₕ, (lower u₃, upper u₃), e_tot); fixed geopotential 10 J kg⁻¹.
const START = (ρ = 2, uₕ = 3, u₃ = (0, 4), e_tot = 100)
const END = (ρ = 3, uₕ = 4, u₃ = (2, 6), e_tot = 110)
const Φ = 10

# Flat Cartesian column with Δz = 1 m, so raw covariant C3 components are in m s⁻¹.
function state(::Type{FT}, s; z_elem) where {FT}
    cspace = CommonSpaces.ColumnSpace(
        FT;
        z_elem,
        z_min = 0,
        z_max = z_elem,
        staggering = CommonSpaces.CellCenter(),
    )
    c = fill(
        (;
            ρ = FT(s.ρ),
            uₕ = Geometry.Covariant12Vector(FT(s.uₕ), FT(0)),
            ρe_tot = FT(s.ρ * s.e_tot),
        ),
        cspace,
    )
    f = fill((; u₃ = Geometry.Covariant3Vector(FT(0))), Spaces.face_space(cspace))
    # Alternate lower/upper face values; a single cell has exactly these two faces.
    for l in 0:z_elem
        Fields.level(f.u₃.components.data.:1, l + half) .= s.u₃[mod(l, 2) + 1]
    end
    return Fields.FieldVector(; c, f)
end

# One ARS343 step with vanishing rates whose final constraint map replaces the state by
# `target`; the observer retains the start and end payloads.
function energy_run(::Type{FT}; z_elem) where {FT}
    Y = state(FT, START; z_elem)
    target = state(FT, END; z_elem)
    constrain_state! = (Y, p, t) -> (Y .= target; nothing)
    odef = CTS.ClimaODEFunction(;
        T_exp_T_lim! = (T_exp, T_lim, U, p, t) ->
            (fill!(T_exp, 0); fill!(T_lim, 0); nothing),
        constrain_state!,
    )
    eobs = AP.EventObserver()
    prob =
        CTS.ODEProblem(AP.wrap_model_hooks(odef, eobs.observer), Y, (FT(0), FT(1)), nothing)
    alg = CTS.IMEXAlgorithm(CTS.ARS343(), CTS.NewtonsMethod(; max_iters = 1))
    integrator = CTS.init(prob, alg; dt = FT(1), solver_observer = eobs)
    # `CTS.init` may apply the final maps to the initial state; the step starts from START.
    integrator.u .= state(FT, START; z_elem)
    CTS.step!(integrator)
    events = eobs.observer.events
    start = only(e for e in events if e.hook == :step_start).data[:u]
    final_end =
        only(e for e in events if e.site == :final_cache && e.phase == :exit).data[:state]
    return start, final_end, events
end

energies(Y) = (
    tot = Y.c.ρe_tot,
    K = Y.c.ρ .* CA.compute_kinetic(Y.c.uₕ, Y.f.u₃),
    Φ = Y.c.ρ .* Φ,
    I = Y.c.ρe_tot .- Y.c.ρ .* CA.compute_kinetic(Y.c.uₕ, Y.f.u₃) .- Y.c.ρ .* Φ,
)
# `compute_kinetic` returns a lazy broadcast; materialize before reading the cell value.
cell(x) = only(parent(Base.Broadcast.materialize(x)))

function check_energy(::Type{FT}) where {FT}
    tol = bound(FT)
    start, final_end, events = energy_run(FT; z_elem = 1)
    K = (
        cell(CA.compute_kinetic(start.c.uₕ, start.f.u₃)),
        cell(CA.compute_kinetic(final_end.c.uₕ, final_end.f.u₃)),
    )
    report("C037 claim 6 $FT K (start, end)", K)
    @test abs(K[1] - 17 // 2) <= tol && abs(K[2] - 18) <= tol
    E0, E1 = energies(start), energies(final_end)
    ΔE = map((a, b) -> cell(b) - cell(a), E0, E1)
    report("C037 claim 6 $FT ΔE (tot, K, Φ, I)", ΔE)
    for (k, start_v, end_v) in
        ((:tot, 200, 330), (:K, 17, 54), (:Φ, 20, 30), (:I, 163, 246))
        @test abs(cell(E0[k]) - start_v) <= tol && abs(cell(E1[k]) - end_v) <= tol
    end
    @test abs(ΔE.tot - 130) <= tol && abs(ΔE.K - 37) <= tol
    @test abs(ΔE.Φ - 10) <= tol && abs(ΔE.I - 83) <= tol
    @test abs(ΔE.tot - (ΔE.K + ΔE.Φ + ΔE.I)) <= tol
    @test abs(ΔE.Φ - Φ * (cell(final_end.c.ρ) - cell(start.c.ρ))) <= tol
    # Fault: squaring the interpolated start vertical velocity, ½(9 + 2²) = 6.5, not 8.5.
    ᶜw = Operators.InterpolateF2C().(start.f.u₃.components.data.:1)
    K_wrong = cell(@. (start.c.uₕ.components.data.:1^2 + ᶜw^2) / 2)
    err = cell(start.c.ρ) * (K_wrong - K[1])
    report("C037 claim 6 $FT fault interpolated-velocity square (ΔE_K error)", err)
    @test abs(err - (-4)) <= tol && abs(err) > tol
end

# Two equal 1 m³ cells; the reconstructed energy channel is start plus the final-map
# increments, compared locally with the retained end payload.
function check_cancellation(::Type{FT}) where {FT}
    tol = bound(FT)
    start, final_end, events = energy_run(FT; z_elem = 2)
    entry =
        only(e for e in events if e.site == :final_constrain && e.phase == :entry).data[:state]
    exit =
        only(e for e in events if e.site == :final_constrain && e.phase == :exit).data[:state]
    recon = start.c.ρe_tot .+ (exit.c.ρe_tot .- entry.c.ρe_tot)
    residual = final_end.c.ρe_tot .- recon
    @test maximum(abs, parent(residual)) <= tol
    # Fault: (+1, −1) J m⁻³ injected into the reconstructed channel only.
    injected = copy(recon)
    Fields.level(injected, 1) .+= 1
    Fields.level(injected, 2) .-= 1
    r = final_end.c.ρe_tot .- injected
    signed, linf, l1 = sum(r), maximum(abs, parent(r)), sum(abs.(r))
    report("C037 claim 6 $FT fault cancellation (signed, L∞, L1)", (signed, linf, l1))
    @test abs(signed) <= tol
    @test abs(linf - 1) <= tol && abs(l1 - 2) <= tol
    @test !(linf <= tol)
end

@testset "C037 event energy" begin
    for FT in (Float32, Float64)
        @testset "claim 6 endpoint energy $FT" check_energy(FT)
        @testset "claim 6 spatial cancellation $FT" check_cancellation(FT)
    end
end
