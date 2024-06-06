include("gmm.jl")

IMAGE_DIR = "images"
image_paths = readdir(IMAGE_DIR) |> x -> filter(x -> x[end-2:end] == "png", x)

OUTPUT_DIR = "outputs-seg4"
images = Dict(zip(image_paths, map(x -> imread("$(IMAGE_DIR)/$(x)"), image_paths)))

# Colorization - qualitative examples

Ns = 2:7

combs = collect(Iterators.product(keys(images), Ns))

ggs = Dict()
lk1 = ReentrantLock()

ggg = Dict()
lk2 = ReentrantLock()

@Threads.threads for (name, N) in combs
    g = gmm_3d_spatial(images[name], N)
    lock(lk1) do
        ggs[(name, N)] = g
    end
end

@Threads.threads for (name, N) in combs
    g = gmm_3d_spatial(greyscale(images[name]), N)
    lock(lk2) do
        ggg[(name, N)] = g
    end
end

exps = collect(Iterators.product(keys(images), keys(images), Ns))

Threads.@threads for (img1, img2, N) in exps
    base1 = img1[1:end-4]
    base2 = img2[1:end-4]
    px1 = greyscale(images[img1])
    px2 = images[img2]

    g1 = ggg[(img1, N)]
    g2 = ggs[(img2, N)]

    m = mapping(g1, g2)
    comp = composite(px1, m, g1, g2)
    imwrite("$(OUTPUT_DIR)/colorizing-$(base1)-from-$(base2)-global-$(N).png", comp)

    m = mapping_spatial(g1, g2)
    comp = composite(px1, m, g1, g2)
    imwrite("$(OUTPUT_DIR)/colorizing-$(base1)-from-$(base2)-spatial-$(N).png", comp)
end

