include("gmm.jl")

IMAGE_DIR = "images"
image_paths = readdir(IMAGE_DIR) |> x -> filter(x -> x[end-2:end] == "png", x)

OUTPUT_DIR = "outputs-seg3"
images = Dict(zip(image_paths, map(x -> imread("$(IMAGE_DIR)/$(x)"), image_paths)))

# Color Transfer - qualitative examples

Ns = 5:10

combs = collect(Iterators.product(keys(images), Ns))

ggs = Dict()

@Threads.threads for (name, N) in combs
    ggs[(name, N)] = gmm_3d_spatial(images[name], N)
end

exps = collect(Iterators.product(keys(images), keys(images), Ns))

Threads.@threads for (img1, img2, N) in exps
    base1 = img1[1:end-4]
    base2 = img2[1:end-4]
    px1 = images[img1]
    px2 = images[img2]

    g1 = ggs[(img1, N)]
    g2 = ggs[(img2, N)]

    m = mapping(g1, g2)
    comp = composite(px1, m, g1, g2)
    imwrite("$(OUTPUT_DIR)/$(base2)-to-$(base1)-global-$(N).png", comp)
end

