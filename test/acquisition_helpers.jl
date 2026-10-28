# Comparison helpers shared by the standalone acquisition tests; `include`d after the
# file's `import ClimaCore: Spaces`. `face_split` reads the including file's global `Y`.

components(x) = (getproperty(x, k) for k in propertynames(x))
exact_equal(a, b) =
    all(parent(x) == parent(y) for (x, y) in zip(components(a), components(b)))
maxdiff(a, b) = maximum(
    maximum(abs, parent(x) .- parent(y)) for (x, y) in zip(components(a), components(b))
)
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
