include("reproducibility.jl")

using SymPy
using QSWalk
using SparseArrays
using LinearAlgebra

x = symbols("x")
w = symbols("w")

for n = 3:3
    # 3 - nodes LQSW example
    a = spzeros(n, n)
    a[1,2:n] .= 1
    @show n
    l = local_lind(a)
    h = a + transpose(a) # a is upper-triangular so this is enough
    s_h = evolve_generator(h, [zeros(size(h)...)])
    s_l = evolve_generator(zeros(size(h)...), l)
    s_h = Matrix(s_h)
    s_l = Matrix(s_l)
    #println(s_h)
    
    #println(s_l)

    #continue
    s = (1-w)*s_h+w*s_l
    det_matrix = s - Diagonal(ones(n^2))*x
    det_s = det_matrix.det()
    det_s_der = diff(det_s, x)
    polynomial = det_s_der.subs(x, 0)
    @show polynomial
    @show solve(polynomial, w)
end