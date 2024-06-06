using Images, FileIO, ImageSegmentation, Statistics, Profile, ShiftedArrays, Random

# reads the pixels of an image into the LAB colorspace
function imread(path)
	img = Lab.(load(path))
	channelview(img)[:, :, :]
end

# writes a LAB image to a path
function imwrite(path, pixels)
    save(path, colorview(Lab, pixels))
end

function greyscale(pixels)
	pixels = copy(pixels)
	pixels[2:3, :, :] .= 1e-9
	pixels
end

mutable struct GMM
	N::Integer
	probs::Array{Float64, 4}   # N x c x h x w
	means::Array{Float64, 2}   # 3 x N
	stds::Array{Float64, 2}    # 3 x N
end

function gmm(c, h, w, N)
	means = zeros(3, N)
	stds = zeros(3, N)
	probs = ones(N, c, h, w) ./ N
	return GMM(N, probs, means, stds)
end

# Generates a new GMM with the same array of source pixels
# Note: no memory is copied - this is to easily generate new GMM states with the same shape
function clone(g::GMM)
	N, c, h, w = size(g.probs)
	gmm(c, h, w, N)
end

function diff(a::Array, b::Array)
    abs(sum(a .- b .+ 1e-9) / sum(a .+ 1e-9))
end

function diff(g1::GMM, g2::GMM)
    return diff(g1.means, g2.means) + diff(g1.stds, g2.stds)
end
#
# given an image of shape (C, H, W), return an iterator over each pixel, with position information
function eachpixel_pos(pixels)
	eachslice(pixels, dims=(2, 3))
end

# given an image of shape (C, H, W), return an iterator over each pixel (without position information)
function eachpixel(pixels)
	c, h, w = size(pixels)
	eachcol(reshape(pixels, (c, h * w)))
end

# a baseline: using k-means clustering instead of GMMs
function gmm_kmeans(pixels, N)
	c, h, w = size(pixels)
	K = kmeans(stack(eachpixel(pixels)), N)
	g = gmm(c, h, w, N)
	g.means = K.centers
	g.stds = hcat([
		stdm(getindex(eachpixel(pixels), findall(==(i), K.assignments)), K.centers[:, i])
		for i in 1:N
	]...)
	g.probs = zeros(size(g.probs))
	for i in 1:N
		idxs = findall(==(i), reshape(K.assignments, h, w))
		v = @view g.probs[i, :, :, :]
		v[:, idxs] .= 1
	end
	g
end

# modifies probs:
#   P(i, x, y) = pixel at (x, y) belongs to ith gaussian
function estep(pixels, g::GMM)
	new_g = clone(g)
	for i in 1:g.N
		new_g.probs[i, :, :, :] = exp.(-((pixels .- g.means[:, i]).^2 ./ (2 .* g.stds[:, i].^2))) .+ 1e-9
	end
	new_g.probs = new_g.probs ./ sum(new_g.probs, dims=1)
	new_g.means = g.means
	new_g.stds = g.stds
	return new_g
end

# modifies means/stds:
#   re-estimates the means/stds of each gaussian
# function mstep(pixels, probs, N)
function mstep(pixels, g)
	new_g = clone(g)
	for i in 1:g.N
		Z = sum(g.probs[i, :, :, :], dims=(2, 3))
		new_g.means[:, i] = sum(g.probs[i, :, :, :] .* pixels, dims=(2, 3)) ./ Z
		new_g.stds[:, i] = (sum(g.probs[i, :, :, :] .* (pixels .- g.means[:, i]).^2, dims=(2, 3)) ./ Z) .^ 0.5
	end
	new_g.probs = g.probs
	return new_g
end

# Can be initialized randomly or using kmeans clustering
function gmm_3d(pixels, N; init=:kmeans, ϵ=0.005, max_t=100, min_t=1)
	c, h, w = size(pixels)
	g = gmm(c, h, w, N)

	if init == :kmeans
		# initializes means as kmean centroids
		K = kmeans(stack(eachpixel(pixels)), N)
		g.means = K.centers
		g.stds = hcat([
			stdm(getindex(eachpixel(pixels), findall(==(i), K.assignments)), K.centers[:, i])
			for i in 1:N
		]...)
		# initialize probabilities
		g = estep(pixels, g)
	else
		# randomly initialize weights
		g.probs = ones(N, c, h, w) ./ N
		# initializes means/stds based on weights
		g = mstep(pixels, g)
	end
	
	t = 0
	while t < max_t
		t += 1
		g1 = estep(pixels, g)
		g2 = mstep(pixels, g1)
		
		d = diff(g, g2)

		g = g2

		@info "Step $(t): d=$(d)"
		if t > min_t && d < ϵ
			break
		end
	end
	@info "EM: $(t) iterations"
	g
end

function show_segments(pixels, g)
	N, c, h, w = size(g.probs)
	masks = [colorview(Gray, mean(g.probs[i, :, :, :], dims=1))[1, :, :] for i in 1:N]
	colors = [fill(colorview(Lab, g.means[:, i])[1], (h, w)) for i in 1:N]
	images = [colorview(Lab, mean(g.probs[i, :, :, :], dims=1) .* pixels) for i in 1:N]
	mosaicview(masks..., colors..., images...; ncol=N, rowmajor=true)
end


function reconstruct(g)
	sum([g.probs[i, :, :, :] .* g.means[:, i] for i in 1:g.N])
end

function estep_spatial(pixels, g)
	new_g = clone(g)
	new_g.probs[:, :, :, :] = g.probs[:, :, :, :]
	for i in 1:g.N
		new_g.probs[i, :, :, :] += exp.(-((pixels .- g.means[:, i]).^2 ./ (2 .* g.stds[:, i].^2))) .+ 1e-9
	end
	new_g.probs = new_g.probs ./ sum(new_g.probs, dims=1)
	new_g.means = g.means
	new_g.stds = g.stds
	return new_g
end

# returns the offsets for each neighbor (all possible offsets except (0, 0))
function neighbors(Ns)
	Iterators.product(-Ns:Ns, -Ns:Ns) |> itr ->
	Iterators.filter(x -> !(x[1] == 0 && x[2]==0), itr)
end

# smooths over the distribution using a bilateral filter
#   Ns  - neighborhood size
#   σd - spatial smoothing factor
#   σg - color smoothing factor
function spatial_smoothing(pixels, g; Ns=2, σd=5, σg=5)
	# construct new probability maps for each possible neighbor offset (with zero-padding)
	N, c, h, w = size(g.probs)
	neighbor_maps = []
	for (dx, dy) in neighbors(Ns)
		shifted = ShiftedArray(pixels, (0, dx, dy), default=0)
		D1 = exp.(-((pixels - shifted).^2) ./ σg)
		D2 = fill(exp(-(dx^2 + dy^2) / σd), size(pixels))
		D = D1 .* D2
		probs_shifted = ShiftedArray(g.probs, (0, 0, dx, dy), default=0)
		res = reshape(D, 1, c, h, w) .* probs_shifted
		push!(neighbor_maps, res)
	end
	result = sum(neighbor_maps) .+ 1e-9
	result = result ./ sum(result, dims=1)
	new_g = clone(g)
	new_g.probs = result
	new_g.stds = g.stds
	new_g.means = g.means
	return new_g
end

function merge(pixels, g; δ=1)
	new_g = clone(g)
	new_g.probs = g.probs
	new_g.means = g.means
	new_g.stds = g.stds
	new_g.N = g.N
	while true
		done = true
		all_pairs = Iterators.product(1:new_g.N, 1:new_g.N) |> x -> Iterators.filter(x -> x[1] < x[2], x)
		for (i, j) in all_pairs
			if mean(abs.(new_g.means[:, i] .- new_g.means[:, j])) < δ
				# merge gaussians i and j
				# merge the probabilities and then re-estimate parameters
				new_g.N -= 1
				new_probs = vcat(new_g.probs[1:(j-1), :, :, :], new_g.probs[(j+1):(new_g.N+1), :, :, :])
				new_probs[i, :, :, :] = new_g.probs[i, :, :, :] .+ new_g.probs[j, :, :, :]
				new_g.probs = new_probs
				new_g = spatial_smoothing(pixels, new_g)
				new_g = mstep(pixels, new_g)
				done = false
				break
			end
		end
		if done
			break
		end
	end
	return new_g
end

function gmm_3d_spatial(pixels, N; init=:kmeans, ϵ=0.005, min_t=5, merge_t=5, max_t=100)
	c, h, w = size(pixels)
	g = gmm(c, h, w, N)

	if init == :kmeans
		# initializes means as kmean centroids
		K = kmeans(stack(eachpixel(pixels)), N)
		g.means = K.centers
		g.stds = hcat([
			stdm(getindex(eachpixel(pixels), findall(==(i), K.assignments)), K.centers[:, i])
			for i in 1:N
		]...)
		# initialize probabilities
		g = estep(pixels, g)
	else
		# randomly initialize weights
		g.probs = ones(N, c, h, w) ./ N
		# initializes means/stds based on weights
		g = mstep(pixels, g)
	end
	
	t = 0
	while t < max_t
		t += 1
		new_g = estep_spatial(pixels, g)
		new_g = spatial_smoothing(pixels, new_g)
		new_g = mstep(pixels, new_g)
		d = diff(g, new_g)
		
		if t > merge_t
			new_g = merge(pixels, new_g)
		end

		g = new_g

		@info "Step $(t): d=$(d), N=$(g.N)"
		if t > min_t && d < ϵ
			break
		end
	end
	@info "EM modified: $(t) iterations"
	g
end

# gaussian mapping function without spatial correspondance
function mapping(g1, g2)
	map = Dict()
	for i in 1:g1.N
		j = argmin(1:g2.N) do j
			if g1.means[1, i] >= g2.means[1, j]
				abs(g1.means[1, i] - g2.means[1, j])
			else
				1e9
			end
		end
		map[i] = j
	end
	return map
end

# gaussian mapping function with spatial correspondance
function mapping_spatial(g1, g2)
	map = Dict()
	for i in 1:g1.N
		j = argmax(1:g2.N) do j
			p1, p2 = sym_paddedviews(0, g1.probs[i, :, :, :], g2.probs[j, :, :, :])
			mean(PaddedViews.no_offset_view(p1) .* PaddedViews.no_offset_view(p2))
		end
		map[i] = j
	end
	return map
end

function composite(pixels, map, g1, g2)
	result = zeros(size(pixels))
	for i in 1:g1.N
		j = map[i]
		result[:, :, :] += g1.probs[i, :, :, :] .* ((g2.stds[:, j] ./ g1.stds[:, i]) .* (pixels .- g1.means[:, i]) .+ g2.means[:, j])
	end
	result
end

# transfer colors from the target to the source image
function transfer(source, target, N)
    gs = gmm_3d_spatial(source, N)
    gt = gmm_3d_spatial(target, N)
    m = mapping_spatial(gs, gt)
    composite(source, m, gs, gt)
end
