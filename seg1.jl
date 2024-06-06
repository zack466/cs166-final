include("gmm.jl")

# Experiments (use kmeans as control)
IMAGE_DIR = "images"
image_paths = readdir(IMAGE_DIR) |> x -> filter(x -> x[end-2:end] == "png", x)

OUTPUT_DIR = "outputs-seg1"
images = Dict(zip(image_paths, map(x -> imread("$(IMAGE_DIR)/$(x)"), image_paths)))

# Segmentation - Effect of number of gaussians
exps = collect(Iterators.product(keys(images), 2:10))

@info "Running with $(Threads.nthreads()) threads"

Threads.@threads for (image_name, N) in exps
    basename = image_name[1:end-4]
    g = gmm_3d_spatial(images[image_name], N)
    output_segments = show_segments(images[image_name], g)
    save("$(OUTPUT_DIR)/$(basename)-$(N)-segments.png", output_segments)
    reconstruction = reconstruct(g)
    imwrite("$(OUTPUT_DIR)/$(basename)-$(N)-reconstruction.png", reconstruction)
end
