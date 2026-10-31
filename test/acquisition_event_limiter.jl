using Test
import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaTimeSteppers as CTS
import ClimaCore: Fields, Geometry, Quadratures, Spaces, Topologies

#=
Production SEM quasimonotone limiter acting on the dry passive tracer `ρq_gas_A`, observed
through the existing solver events (`:stage_lim`, `:final_lim`, `AssemblyInputs`) and hook
wrappers; standalone, not part of runtests.jl. Needs a ClimaTimeSteppers with labelled solver
events. One ARS343 step of an active case and of a matched control with the limiter
disabled, in Float32 and Float64. The chemistry stays a no-op: `Musica` is not imported.

The operands and pre-limiter states come from the active records, which equal the control's
through `exact_equal(control_entry, entry)`.

- Tracer: `q = q₀ + a 𝟏[(i, j) = (3, 2)]` in slab (h*, v*), a = 1/16, q₀ = a/64 = 1/1024.
- Slab: `select_slab` on the stage-1 tendency state gives (h*, v*) = (1, 7) in both precisions;
  (5, 7) ties by symmetry and the lowest (h, v) wins. No limiter output enters the rule.
- Target node (2, 2), with C and S of `target_terms`, T_lim = -aC - q₀S:
    Float32: C = 4.42377e-6, S = 1.28002e-7, q₀|S| = 1.25e-10 ≤ aC/2 = 1.38e-7
    Float64: C = 4.42377e-6, S = 1.27999e-7, q₀|S| = 1.25e-10 ≤ aC/2 = 1.38e-7
- Bounds (reconstructed, no event emits them; `reconstructed_bounds`): target slab
  [q_min, q_max] = [q₀, q₀ + a] = [9.765625e-4, 6.34765625e-2] with the pre-limiter slab
  mean 1.22094e-2 inside it, so the mean does not widen them.
- First-pass clipping set of the target slab: no upper clips; lower clips
    Float32: (1,1) (2,1) (3,1) (2,2) (1,3) (2,4) (3,4) (4,4)
    Float64: the same plus (1,4)
  Redistribution values are not asserted, nor is (2, 2) the only clipped node. Outside this
  slab the q₀S term also triggers clipping, so nowhere else is activity asserted.
- Activity margin, with γ = a_exp[2,1]; the values are at the configured dt = 1. Predicted
  undershoot at the target U = dt γ (aC + q₀S) ≥ dt γ aC/2 = 6.03e-8.
  Size of the background correction in U: dt γ q₀|S| = 5.45e-11, 4.5e-4 of U.
  All nodal operands of the clipped target lie near M = ρ₂₂q₀ = 3.619e-4: the pre-limiter
  value and the clipped value `ρ q_min`. `ρ q_max` enters only nodes clipped to the upper
  bound, and redistribution towards q_max only raises the target, so it stays out of M.
  With u = eps(FT)/2 and γ_n = nu/(1 - nu):
    τ = 8 eps(FT) M.
    E_A = γ_40 dt γ T_abs + γ_8 M: the first term bounds the tendency formation (at most 40
    rounded operations, scaled by the sum T_abs of the absolute split terms, 2.96e-7), the
    second the initialization, division, stage accumulation and clipping at the target.
    Float32: τ = 3.45e-10, E_A = 1.73e-10, U/(τ + E_A) = 232.7
    Float64: τ = 6.43e-19, E_A = 3.22e-19, U/(τ + E_A) = 1.25e11
  The undershoot and the correction at the target are each asserted to be at least
  `MIN_MARGIN` = 16 times τ + E_A in both precisions. τ covers round-off only.
- Tolerances: reconstructed bounds 16 eps(FT) relative to the bound (clipping and rounding
  of the mean, a few rounded operations); slab tracer mass 64 eps(FT) relative to the slab
  mass (limiter `rtol = eps`, 16-term summations, node write rounding).
- Final-limiter activity is recorded, not asserted.
=#
println("CTS from ", pathof(CTS))

include("acquisition_helpers.jl")
include("solver_event_consumer.jl")
report(name, value) = println(rpad(name, 60), " = ", value)

const SPIKE_A = 1 / 16
const BACKGROUND_Q0 = SPIKE_A / 64
const ARS343_GAMMA = 0.4358665215084590
const SPIKE_SLAB = (1, 7)
const CLIPPED_LOWER = Dict(
    Float32 => [(1, 1), (2, 1), (3, 1), (2, 2), (1, 3), (2, 4), (3, 4), (4, 4)],
    Float64 => [(1, 1), (2, 1), (3, 1), (2, 2), (1, 3), (1, 4), (2, 4), (3, 4), (4, 4)],
)
const MIN_MARGIN = 16

function limiter_config(::Type{FT}, sem) where {FT}
    return CA.AtmosConfig(
        Dict(
            "config" => "sphere",
            "initial_condition" => "DryBaroclinicWave",
            "h_elem" => 2,
            "nh_poly" => 3,
            "z_elem" => 10,
            "dt" => "1secs",
            "t_end" => "2secs",
            "FLOAT_TYPE" => string(FT),
            "log_progress" => false,
            "enable_diagnostics" => false,
            "output_default_diagnostics" => false,
            "disable_surface_flux_tendency" => true,
            "output_dir" => mktempdir(),
            "microphysics_model" => "dry",
            "chemistry_model" => "passive",
            "hyperdiff" => nothing,
            "tracer_nonnegativity_method" => nothing,
            "prescribed_flow" => nothing,
            "apply_sem_quasimonotone_limiter" => sem,
        );
        job_id = "acquisition_event_limiter",
    )
end

# The tracer density of a positive background plus, for `spike = (h, v)`, an interior
# node (3, 2) of that slab. Native indexing: parent(field)[v, i, j, 1, h].
function set_tracer!(Y, ::Type{FT}, spike) where {FT}
    Y.c.ρq_gas_A .= Y.c.ρ .* FT(BACKGROUND_Q0)
    if !isnothing(spike)
        (h, v) = spike
        parent(Y.c.ρq_gas_A)[v, 3, 2, 1, h] =
            parent(Y.c.ρ)[v, 3, 2, 1, h] * FT(BACKGROUND_Q0 + SPIKE_A)
    end
    return Y
end

# The tracer is written to Y before build_cache, so the stages of AtmosSimulation are
# spelled out here.
function assemble(::Type{FT}, sem) where {FT}
    config = limiter_config(FT, sem)
    pa = config.parsed_args
    params = CA.ClimaAtmosParameters(config)
    setup = CA.get_setup_type(pa, CA.Parameters.thermodynamics_params(params))
    grid = CA.get_grid(pa, params, config.comms_ctx)
    model = CA.get_atmos(config, params, grid; setup_type = setup)
    return (; pa, params, model, Y = CA.initial_state(model))
end

function observed_step(::Type{FT}, parts, spike) where {FT}
    (; pa, params, model, Y) = parts
    set_tracer!(Y, FT, spike)
    initial_tracer = copy(Y.c.ρq_gas_A)
    start_date = CA.parse_date(pa["start_date"])
    dt, t_start, t_end =
        CA.convert_time_args(pa["dt"], pa["t_start"], pa["t_end"], start_date)
    p = CA.build_cache(Y, model, params, dt, start_date, nothing)
    args, kwargs = CA.args_integrator(
        Y, p, (t_start, t_end), CA.ode_configuration(FT, pa), CTS.CallbackSet(),
        CA.jacobian_from_parsed_args(pa), pa["debug_jacobian"], model.prescribed_flow,
        dt, pa["update_cache_every"], pa["update_constrain_state_every"],
    )
    (prob, alg) = args
    rec = EventRecorder()
    integrator = CTS.init(
        CTS.ODEProblem(wrap_hooks(prob.f, rec), prob.u0, prob.tspan, prob.p),
        alg;
        kwargs...,
        solver_observer = rec,
    )
    CTS.step!(integrator)
    return (; integrator, records = rec.records, initial_tracer, p)
end

record_at(es, kind, site, stage, phase) = only(
    e for e in es if
    e.kind == kind && e.site == site && e.stage == stage && e.phase == phase
)
operation(es, site, stage, phase) = record_at(es, :Operation, site, stage, phase)
hook(es, site, stage, phase) = record_at(es, :hook, site, stage, phase)
const LIMITER_CALLS = [(:stage_lim, 2), (:stage_lim, 3), (:stage_lim, 4), (:final_lim, 0)]

# Widened flux operands of the split divergence, from the state of the stage-1 tendency.
function flux_operands(U)
    FT = eltype(parent(U.c))
    space = axes(U.c)
    lg = Fields.local_geometry_field(space)
    u = Geometry.Covariant123Vector.(U.c.uₕ)
    ρ = Float64.(parent(U.c.ρ))
    J = Float64.(parent(lg.J))
    G¹ = ρ .* J .* Float64.(parent(Geometry.contravariant1.(u, lg)))
    G² = ρ .* J .* Float64.(parent(Geometry.contravariant2.(u, lg)))
    quadrature = Spaces.quadrature_style(Spaces.horizontal_space(space))
    D = Float64.(Quadratures.differentiation_matrix(FT, quadrature))
    return (; ρ, J, G¹, G², D)
end

# `C` and `S` of the hand derivation at node (2, 2) of slab (h, v).
function target_terms(op, v, h)
    (; ρ, J, G¹, G², D) = op
    J₂₂ = J[v, 2, 2, 1, h]
    C = D[2, 3] * (G¹[v, 2, 2, 1, h] + G¹[v, 3, 2, 1, h]) / (2 * J₂₂)
    S = sum(D[2, k] * (G¹[v, k, 2, 1, h] + G²[v, 2, k, 1, h]) for k in 1:4) / J₂₂
    return (; C, S, ρ₂₂ = ρ[v, 2, 2, 1, h])
end

# Split-form tracer tendency at (2, 2) from widened nodal concentrations `q`, and the sum
# of the absolute values of its terms, which scales its rounding error.
function split_tendency(op, q, v, h)
    (; J, G¹, G², D) = op
    T = 0.0
    T_abs = 0.0
    for k in 1:4
        x =
            D[2, k] * (G¹[v, k, 2, 1, h] + G¹[v, 2, 2, 1, h]) *
            (q[v, k, 2, 1, h] + q[v, 2, 2, 1, h]) / 2
        y =
            D[2, k] * (G²[v, 2, k, 1, h] + G²[v, 2, 2, 1, h]) *
            (q[v, 2, k, 1, h] + q[v, 2, 2, 1, h]) / 2
        T -= x + y
        T_abs += abs(x) + abs(y)
    end
    return (; T = T / J[v, 2, 2, 1, h], T_abs = T_abs / J[v, 2, 2, 1, h])
end

# The selection rule, on operands only: the largest C/ρ₂₂ among slabs with
# q₀|S| ≤ aC/2 and C > 0; values within 0.1 % of the largest tie, and the lowest (h, v)
# of the ties wins, so that both precisions select one slab.
function select_slab(op)
    (Nv, _, _, _, Nh) = size(op.ρ)
    scored = [(h, v, target_terms(op, v, h)) for h in 1:Nh for v in 1:Nv]
    admissible = [
        (h, v, t.C / t.ρ₂₂) for (h, v, t) in scored if
        t.C > 0 && BACKGROUND_Q0 * abs(t.S) ≤ SPIKE_A * t.C / 2
    ]
    largest = maximum(last, admissible)
    ties = [(h, v) for (h, v, r) in admissible if r ≥ (1 - 1e-3) * largest]
    return minimum(ties), largest, admissible
end

# Reconstructed limiter bounds (not emitted by any event): nodal concentration extrema of
# the reference over each slab and its same-level neighbours, widened to include the
# pre-limiter slab mean. `ClimaCore.Limiters` applies them per slab.
function reconstructed_bounds(ref, pre)
    space = axes(pre.c)
    topology = Spaces.topology(Spaces.horizontal_space(space))
    WJ = Float64.(parent(Fields.local_geometry_field(space).WJ))
    q = parent(ref.c.ρq_gas_A) ./ parent(ref.c.ρ)
    (Nv, _, _, _, Nh) = size(q)
    lo = zeros(Float64, Nv, Nh)
    hi = zeros(Float64, Nv, Nh)
    for h in 1:Nh, v in 1:Nv
        slabs = (h, Topologies.local_neighboring_elements(topology, h)...)
        lo[v, h] = minimum(minimum(@view q[v, :, :, 1, n]) for n in slabs)
        hi[v, h] = maximum(maximum(@view q[v, :, :, 1, n]) for n in slabs)
        m = slab_mean(WJ, pre, v, h)
        lo[v, h] = min(lo[v, h], m)
        hi[v, h] = max(hi[v, h], m)
    end
    return (; lo, hi, WJ)
end

slab_mass(WJ, x, v, h) =
    sum(WJ[v, i, j, 1, h] * Float64(x[v, i, j, 1, h]) for i in 1:4, j in 1:4)
slab_mean(WJ, state, v, h) =
    slab_mass(WJ, parent(state.c.ρq_gas_A), v, h) / slab_mass(WJ, parent(state.c.ρ), v, h)

# First-pass clipping set of slab (h, v), from pre-limiter values and bounds only.
function clipping_set(pre, bounds, v, h)
    ρq = Float64.(parent(pre.c.ρq_gas_A))
    ρ = Float64.(parent(pre.c.ρ))
    q(i, j) = ρq[v, i, j, 1, h] / ρ[v, i, j, 1, h]
    lower = [(i, j) for i in 1:4, j in 1:4 if q(i, j) < bounds.lo[v, h]]
    upper = [(i, j) for i in 1:4, j in 1:4 if q(i, j) > bounds.hi[v, h]]
    return (; lower, upper)
end

dry_equal(a, b) =
    parent(a.c.ρ) == parent(b.c.ρ) && parent(a.c.ρe_tot) == parent(b.c.ρe_tot) &&
    parent(a.c.uₕ) == parent(b.c.uₕ) && parent(a.f.u₃) == parent(b.f.u₃)

function check_setup(::Type{FT}, active, control) where {FT}
    (h, v) = SPIKE_SLAB
    @test CA.is_tracer_var(:ρq_gas_A)
    @test !any(id -> id.name == "Musica", keys(Base.loaded_modules))
    @test :ρq_tot ∉ propertynames(active.integrator.u.c)
    p_active, p_control = active.p, control.p
    @test p_active.numerics.sem_quasimonotone_limiter !== nothing
    @test p_control.numerics.sem_quasimonotone_limiter === nothing
    for p in (p_active, p_control)
        @test p.numerics.vertical_water_borrowing_limiter === nothing
        @test p.atmos.microphysics_model isa CA.DryModel
        @test p.atmos.turbconv_model === nothing
        @test p.atmos.hyperdiff === nothing
    end
    tableau = active.integrator.cache.tableau
    @test active.integrator.alg.name isa CTS.ARS343
    @test tableau.a_exp[2, 1] ≈ ARS343_GAMMA
    for run in (active, control)
        es = run.records
        Y_n = only(e for e in es if e.kind == :StepStart).data[:u]
        U₁ = hook(es, :T_exp_T_lim!, 1, :entry).data[:state]
        # The setup holds the test's tracer, and the stage-2 entry keeps the density.
        @test parent(run.initial_tracer) == parent(Y_n.c.ρq_gas_A)
        @test parent(operation(es, :stage_lim, 2, :entry).data[:state].c.ρ) ==
              parent(Y_n.c.ρ)
        # The operands come from the stage-1 tendency state.
        report("$FT max|U₁ - StepStart|", maxdiff(U₁, Y_n))
        @test maxdiff(U₁, Y_n) == 0
    end
    @test parent(active.initial_tracer) == parent(control.initial_tracer)
    ρq₀ = parent(active.initial_tracer)
    Y_n = only(e for e in active.records if e.kind == :StepStart).data[:u]
    q₀ = ρq₀ ./ parent(Y_n.c.ρ)
    @test size(ρq₀) == (10, 4, 4, 1, 24)
    @test all(>(0), ρq₀)
    @test count(q₀ .> 2 * FT(BACKGROUND_Q0)) == 1
    # The product `ρ q` written by the setup and this division each round once.
    @test isapprox(q₀[v, 3, 2, 1, h], FT(BACKGROUND_Q0 + SPIKE_A); rtol = 2 * eps(FT))
end

function check_events(::Type{FT}, active, control) where {FT}
    roles(es) = [e.role for e in es if e.site in (:stage_lim, :final_lim)]
    @test roles(active.records) == roles(control.records)
    for run in (active, control)
        es = run.records
        Y_n = only(e for e in es if e.kind == :StepStart).data[:u]
        limiter_ops = [
            (e.site, e.stage, e.phase) for
            e in es if e.kind == :Operation && e.site in (:stage_lim, :final_lim)
        ]
        @test limiter_ops ==
              [(s, i, p) for (s, i) in LIMITER_CALLS for p in (:entry, :exit)]
        for (site, stage) in LIMITER_CALLS
            entry = operation(es, site, stage, :entry)
            exit = operation(es, site, stage, :exit)
            @test entry.role == exit.role !== nothing
            @test exact_equal(entry.data[:extra], Y_n)
            @test exact_equal(exit.data[:extra], Y_n)
            @test dry_equal(entry.data[:state], exit.data[:state])
        end
        report("$FT limiter roles", unique(roles(es)))
    end
end

function check_activity(::Type{FT}, active, control) where {FT}
    (h, v) = SPIKE_SLAB
    es = active.records
    Y_n = only(e for e in es if e.kind == :StepStart).data[:u]
    U₁ = hook(es, :T_exp_T_lim!, 1, :entry).data[:state]
    T_lim₁ = hook(es, :T_exp_T_lim!, 1, :exit).data[:T_lim]
    entry = operation(es, :stage_lim, 2, :entry).data[:state]
    exit = operation(es, :stage_lim, 2, :exit).data[:state]
    control_entry = operation(control.records, :stage_lim, 2, :entry).data[:state]
    control_exit = operation(control.records, :stage_lim, 2, :exit).data[:state]

    op = flux_operands(U₁)
    @test op.D[2, 3] ≈ sqrt(FT(5)) / 2
    @test first(select_slab(op)) == SPIKE_SLAB
    t = target_terms(op, v, h)
    @test t.C > 0
    @test BACKGROUND_Q0 * abs(t.S) ≤ SPIKE_A * t.C / 2

    u = Float64(eps(FT)) / 2
    γ_n(n) = n * u / (1 - n * u)
    dt = Float64(float(active.integrator.dt))
    dtγ = dt * Float64(active.integrator.cache.tableau.a_exp[2, 1])
    q = Float64.(parent(U₁.c.ρq_gas_A)) ./ Float64.(parent(U₁.c.ρ))
    predicted = split_tendency(op, q, v, h)
    M = t.ρ₂₂ * BACKGROUND_Q0
    τ = 8 * Float64(eps(FT)) * M
    E_A = γ_n(40) * dtγ * predicted.T_abs + γ_n(8) * M
    recorded = Float64(parent(T_lim₁.c.ρq_gas_A)[v, 2, 2, 1, h])
    report("$FT T_lim target predicted, recorded", (predicted.T, recorded))
    @test abs(predicted.T - recorded) ≤ γ_n(40) * predicted.T_abs

    bounds = reconstructed_bounds(Y_n, entry)
    q_min, q_max = bounds.lo[v, h], bounds.hi[v, h]
    q_bound_tol = 16 * eps(FT)
    @test isapprox(q_min, BACKGROUND_Q0; rtol = q_bound_tol)
    @test isapprox(q_max, BACKGROUND_Q0 + SPIKE_A; rtol = q_bound_tol)
    @test q_min < slab_mean(bounds.WJ, entry, v, h) < q_max

    pre = Float64(parent(entry.c.ρq_gas_A)[v, 2, 2, 1, h])
    post = Float64(parent(exit.c.ρq_gas_A)[v, 2, 2, 1, h])
    undershoot = t.ρ₂₂ * q_min - pre
    predicted_undershoot = dtγ * (SPIKE_A * t.C + BACKGROUND_Q0 * t.S)
    correction = post - pre
    margin = undershoot / (τ + E_A)
    report("$FT undershoot observed, predicted", (undershoot, predicted_undershoot))
    report("$FT τ, E_A", (τ, E_A))
    report("$FT background correction dtγq₀|S|", dtγ * BACKGROUND_Q0 * abs(t.S))
    report("$FT target correction, margin to τ + E_A", (correction, correction / (τ + E_A)))
    report("$FT undershoot margin", margin)
    @test abs(undershoot - predicted_undershoot) ≤ E_A
    @test margin ≥ MIN_MARGIN
    @test correction ≥ MIN_MARGIN * (τ + E_A)
    @test correction ≥ undershoot - E_A

    # The control enters the limiter with the same state and leaves the target at another
    # value; the testset "control is an exact no-op" covers the control's limiter calls.
    @test exact_equal(control_entry, entry)
    @test parent(control_exit.c.ρq_gas_A)[v, 2, 2, 1, h] !=
          parent(exit.c.ρq_gas_A)[v, 2, 2, 1, h]

    clipped = clipping_set(entry, bounds, v, h)
    report("$FT reconstructed first-pass clipping set", clipped)
    @test clipped.lower == CLIPPED_LOWER[FT]
    @test isempty(clipped.upper)
end

function check_bounds_and_mass(::Type{FT}, active) where {FT}
    es = active.records
    (Nv, _, _, _, Nh) = size(parent(active.initial_tracer))
    for (site, stage) in LIMITER_CALLS
        entry = operation(es, site, stage, :entry)
        pre = entry.data[:state]
        post = operation(es, site, stage, :exit).data[:state]
        bounds = reconstructed_bounds(entry.data[:extra], pre)
        (; lo, hi, WJ) = bounds
        ρq⁺ = Float64.(parent(post.c.ρq_gas_A))
        q⁺ = ρq⁺ ./ Float64.(parent(post.c.ρ))
        worst_bound = 0.0
        worst_mass = 0.0
        smallest_inventory = Inf
        for h in 1:Nh, v in 1:Nv
            slab_q = @view q⁺[v, :, :, 1, h]
            worst_bound = max(
                worst_bound,
                (lo[v, h] - minimum(slab_q)) / abs(lo[v, h]),
                (maximum(slab_q) - hi[v, h]) / abs(hi[v, h]),
            )
            before = slab_mass(WJ, parent(pre.c.ρq_gas_A), v, h)
            after = slab_mass(WJ, ρq⁺, v, h)
            worst_mass = max(worst_mass, abs(after - before) / before)
            smallest_inventory = min(smallest_inventory, before)
        end
        report("$FT $site $stage bound excess / eps", worst_bound / eps(FT))
        report("$FT $site $stage mass change / eps", worst_mass / eps(FT))
        @test smallest_inventory > 0
        @test worst_bound ≤ 16 * eps(FT)
        @test worst_mass ≤ 64 * eps(FT)
    end
end

function check_final_limiter(::Type{FT}, active, control) where {FT}
    for run in (active, control)
        es = run.records
        a = only(e for e in es if e.kind == :AssemblyInputs).data
        entry = operation(es, :final_lim, 0, :entry).data[:state]
        exit = operation(es, :final_lim, 0, :exit).data[:state]
        rebuilt = copy(a[:u].c.ρq_gas_A)
        dt = FT(float(a[:dt]))
        for (j, T_lim_j) in a[:T_lim]
            weight = a[:b_exp][j]
            @. rebuilt += dt * weight * T_lim_j.c.ρq_gas_A
        end
        tracer_scale = maximum(abs, parent(entry.c.ρq_gas_A))
        residual = maximum(abs, parent(entry.c.ρq_gas_A) .- parent(rebuilt))
        @test residual ≤ 64 * eps(FT) * tracer_scale
        correction =
            Float64.(parent(exit.c.ρq_gas_A)) .- Float64.(parent(entry.c.ρq_gas_A))
        (h, v) = SPIKE_SLAB
        report(
            "$FT final correction max|L⁺-L⁻|, at the target",
            (maximum(abs, correction), correction[v, 2, 2, 1, h]),
        )
    end
end

@testset "SEM limiter on a dry passive tracer, $FT" for FT in (Float32, Float64)
    active, control = nothing, nothing
    @testset "setup and identities" begin
        active = observed_step(FT, assemble(FT, true), SPIKE_SLAB)
        control = observed_step(FT, assemble(FT, false), SPIKE_SLAB)
        check_setup(FT, active, control)
    end
    # A throw in the setup is already reported; the remaining checks need both runs.
    (isnothing(active) || isnothing(control)) && continue
    @testset "limiter event structure" begin
        check_events(FT, active, control)
    end
    @testset "stage-2 activity at the target against the control" begin
        check_activity(FT, active, control)
    end
    @testset "reconstructed bounds and slab mass" begin
        check_bounds_and_mass(FT, active)
    end
    @testset "control is an exact no-op" begin
        for (site, stage) in LIMITER_CALLS
            @test exact_equal(
                operation(control.records, site, stage, :entry).data[:state],
                operation(control.records, site, stage, :exit).data[:state],
            )
        end
    end
    @testset "final limiter operands, recorded only" begin
        check_final_limiter(FT, active, control)
    end
end
