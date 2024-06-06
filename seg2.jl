include("gmm.jl")

# Experiments (use kmeans as control)
IMAGE_DIR = "images"
image_paths = readdir(IMAGE_DIR) |> x -> filter(x -> x[end-2:end] == "png", x)

OUTPUT_DIR = "outputs-seg2"
images = Dict(zip(image_paths, map(x -> imread("$(IMAGE_DIR)/$(x)"), image_paths)))

exps = collect(Iterators.product(keys(images), 3:8))

# Segmentation - Does the spatial filtering help?
Threads.@threads for (image_name, N) in exps
    basename = image_name[1:end-4]

    g = gmm_3d_spatial(images[image_name], N)
    output_segments = show_segments(images[image_name], g)
    save("$(OUTPUT_DIR)/$(basename)-$(N)-spatial-segments.png", output_segments)
    reconstruction = reconstruct(g)
    imwrite("$(OUTPUT_DIR)/$(basename)-$(N)-spatial-reconstruction.png", reconstruction)

    g = gmm_3d(images[image_name], N)
    output_segments = show_segments(images[image_name], g)
    save("$(OUTPUT_DIR)/$(basename)-$(N)-3d-segments.png", output_segments)
    reconstruction = reconstruct(g)
    imwrite("$(OUTPUT_DIR)/$(basename)-$(N)-3d-reconstruction.png", reconstruction)

    g = gmm_kmeans(images[image_name], N)
    output_segments = show_segments(images[image_name], g)
    save("$(OUTPUT_DIR)/$(basename)-$(N)-kmeans-segments.png", output_segments)
    reconstruction = reconstruct(g)
    imwrite("$(OUTPUT_DIR)/$(basename)-$(N)-kmeans-reconstruction.png", reconstruction)
end
