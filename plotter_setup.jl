using Pkg
Pkg.activate(".")

using PyPlot
using NPZ
rc("text", usetex=true)
rc("font", family="serif")
rc(Dict("savefig.bbox_inches" => "tight"))
rc(Dict("savefig.metadata" => Dict("CreationDate" => nothing)))


linreg(x, y) = hcat(fill!(similar(x), 1), x) \ y

function scaling_exponent_generator(data_x::Vector,
                                    data_y::Vector,
                                    batch::Int=11)
   @assert length(data_y) == length(data_x)
   @assert length(data_x) >= batch
   @assert isodd(batch)

   result = Float64[]
   #println(result)
   for i=1:(length(data_y)-batch+1)
      #println(linreg(data_x[i:i+batch-1], data_y[i:i+batch-1]))
      push!(result, linreg(data_x[i:i+batch-1], data_y[i:i+batch-1])[2])
   end
   data_x[div(batch, 2)+1:end-div(batch, 2)], result
end

#println(collect(1:20.))
#println(2 .+ 3. .* collect(1:20))
#scaling_exponent_generator( collect(1:20.) , 2 .+ 3 .* collect(1:20.))
