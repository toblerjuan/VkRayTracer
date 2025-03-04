#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "lib/tiny_obj_loader.h"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <optional>
#include <array>
#include <chrono>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const size_t GEOMETRY_COUNT = 3;
const size_t OBJECT_COUNT = 4;

const std::string MODEL_PATH = "models/bunny.obj";
const std::string R_MODEL_PATH = "models/wavy.obj";

const uint32_t shaderCount = 3;
const uint32_t groupCount = 3;

// Specifying what validation layers we want to use
const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    // Ray tracing extensions (according to RTG II)
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
    // Ray tracing validation layers
    // VK_NV_RAY_TRACING_VALIDATION_EXTENSION_NAME
};

// TMU: If we launch in debug mode, we will use val layers, otherwise not
#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif
//const bool enableValidationLayers = false;

struct Vertex {
    alignas(16) glm::vec3 pos;
    alignas(16) glm::vec3 normal;

    bool operator==(const Vertex &other) const
    {
        return pos == other.pos && normal == other.normal;   
    }

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 1> 
        getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 1> 
            attributeDescriptions{};
        // vertex positions
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);
        // vertex colors
        return attributeDescriptions;
    }
};

// Beyond my comprehension ¯\_(Ü)_/¯
namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const &vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^
                (hash<glm::vec3>()(vertex.normal) << 1)) >> 1);
        }
    };
}

struct Geometry 
{
    static size_t nextFirstVtx;
    static uint32_t nextPrimOffset; // in bytes

    // format for vertex: {pos.x, pos.y, pos.z, 0, nrml.t, nrml.b, nrml.n, 0}
    static std::vector<Vertex> vertices;
    static VkBuffer vtxBuf;
    static VkDeviceMemory vtxBufMem;

    static std::vector<uint32_t> indices;
    static VkBuffer idxBuf;
    static VkDeviceMemory idxBufMem;

    VkBuffer blasBuf;
    VkDeviceMemory blasBufMem;
    VkAccelerationStructureKHR blas;

    size_t firstVtx; // array position of first vertex of geometry
    uint32_t primCount; // how many primitives geometry has (#idx / 3)
    uint32_t primOffset; // how many prims to skip in idx array IN BYTES

    Geometry(
        std::vector<Vertex> geomVertices, 
        std::vector<uint32_t> geomIndices) 
    {
        firstVtx = nextFirstVtx;
        primCount = static_cast<uint32_t>(geomIndices.size() / 3);
        primOffset = nextPrimOffset;

        nextFirstVtx += geomVertices.size();
        nextPrimOffset += 3 * sizeof(uint32_t) * primCount;

        vertices.insert(vertices.end(), geomVertices.begin(), 
            geomVertices.end());
        indices.insert(indices.end(), geomIndices.begin(),
            geomIndices.end());
    }
};

size_t Geometry::nextFirstVtx = 0;
uint32_t Geometry::nextPrimOffset = 0;

std::vector<Vertex> Geometry::vertices;
VkBuffer Geometry::vtxBuf;
VkDeviceMemory Geometry::vtxBufMem;

std::vector<uint32_t> Geometry::indices;
VkBuffer Geometry::idxBuf;
VkDeviceMemory Geometry::idxBufMem;

struct Material
{
    alignas(16) glm::vec3 color;
    alignas(4) float reflectivity;
    alignas(4) float refractivity;
    alignas(4) float indexOfRefraction;
    alignas(4) float shininess;
};

struct Object 
{
    Geometry *geometry;
    glm::mat4 model_transform;

    Material material;
};

struct UniformBufferObject {
    glm::mat4 view;
    glm::mat4 view_inv;
    glm::mat4 proj;
    glm::mat4 proj_inv;
};

// Yoinked this from vulkan tutorial. Is it possible to move this into the other
// function loader function thing?
VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance, 
    const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo, 
    VkAllocationCallbacks *pAllocator, 
    VkDebugUtilsMessengerEXT *pDebugMessenger) 
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}
void DestroyDebugUtilsMessengerEXT(
    VkInstance instance, 
    VkDebugUtilsMessengerEXT debugMessenger, 
    const VkAllocationCallbacks *pAllocator) 
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

class HelloTriangleApplication 
{
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        //cleanup(); // for now...
    }

private:
    GLFWwindow *window;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSurfaceKHR surface;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    int MAX_FRAMES_IN_FLIGHT = -1;

    // uniforms
    const uint32_t BIND_IMG = 1;
    const uint32_t BIND_TLAS = 0;
    const uint32_t BIND_MTX = 2;
    const uint32_t BIND_VERTS = 3;
    const uint32_t BIND_INDICES = 4;
    const uint32_t BIND_FIRST_VTX = 5;
    const uint32_t BIND_PRIM_OFFSET = 6;
    const uint32_t BIND_MAT = 7;

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    VkPipelineLayout rtPipelineLayout;
    VkPipeline rtPipeline;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;

    std::vector<Geometry> geometries;
    std::vector<Object> objects;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void *> uniformBuffersMapped;
    // Buffers that tell shaders how to index into the shared vertex buffer
    VkBuffer firstVerticesBuffer;
    VkDeviceMemory firstVerticesBufferMemory;
    VkBuffer primOffsetsBuffer;
    VkDeviceMemory primOffsetsBufferMemory;
    VkBuffer materialsBuffer;
    VkDeviceMemory materialsBufferMemory;

    VkBuffer TLAS_buffer;
    VkDeviceMemory TLAS_bufferMemory;
    // Should be able to get rid of these scratch buffers and have them locally
    // inside the function
    VkBuffer TLAS_scratchBuffer;
    VkDeviceMemory TLAS_scratchBufferMemory;

    std::vector<VkCommandBuffer> blasCmdBufs;
    VkCommandBuffer tlasCmdBuf;

    VkBuffer instancesBuffer; // stores blas instances
    VkDeviceMemory instancesBufferMemory;
    VkBuffer transformBuffer;
    VkDeviceMemory transformBufferMemory;

    VkAccelerationStructureKHR tlas;
    
    VkBuffer SBT_buffer;
    VkDeviceMemory SBT_bufferMemory;

    VkBuffer rgenBuffer;
    VkDeviceMemory rgenBufferMemory;
    VkBuffer hitBuffer;
    VkDeviceMemory hitBufferMemory;
    VkBuffer missBuffer;
    VkDeviceMemory missBufferMemory;

    uint32_t groupSizeAligned;

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtPipelineProperties{};

    // asdf
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
    PFN_vkGetAccelerationStructureBuildSizesKHR 
        vkGetAccelerationStructureBuildSizesKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR
        vkCmdBuildAccelerationStructuresKHR;
    PFN_vkGetAccelerationStructureDeviceAddressKHR
        vkGetAccelerationStructureDeviceAddressKHR;
    PFN_vkGetRayTracingShaderGroupHandlesKHR 
        vkGetRayTracingShaderGroupHandlesKHR;
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR;
    PFN_vkCreateRayTracingPipelinesKHR  vkCreateRayTracingPipelinesKHR;
    PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT;


    void initWindow() 
    {
        glfwInit();
        // tell glfw we are not using openGL
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        // window resizing needs special care
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() 
    {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        loadRayTracingFunctions();
        createSwapChain();
        createImageViews(); // maybe rename to swapchainImageViews
        //createRenderPass();
        createDescriptorSetLayoutRT();
        createRayTracingPipeline();
        createSBT(); // after rt pipeline
        //createFramebuffers();
        createCommandPool();
        createObjects();
        createUniformBuffers();
        buildAS();
        createDescriptorPoolRT();
        createDescriptorSetsRT();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop() 
    {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        // Program crashes when closed. This is because commands in drawFrame
        // are asynchronous, meaning that when we exit the main loop and begin
        // cleaning up, there may still be drawing/presentation operations
        // happening in the background. Thus, we wait for logical device to
        // become idle before we start cleaning up:
        vkDeviceWaitIdle(device);
    }


    // stupid question: why do we need to destroy all this stuff? won't it
    // happen automatically when we close the program?
    void cleanup() 
    {
        cleanupSwapChain();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        /*
        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
        */

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }
        vkDestroyCommandPool(device, commandPool, nullptr);
        //vkDestroyPipeline(device, graphicsPipeline, nullptr);
        //vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        //vkDestroyRenderPass(device, renderPass, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyDevice(device, nullptr);
        if (enableValidationLayers)
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        vkDestroyInstance(instance, nullptr);
        glfwDestroyWindow(window);
        glfwTerminate();
    }


    struct QueueFamilyIndices
    {
        // TMU: we are looking for a queue that does graphics, a queue that
        // can present (i.e. display to a screen), and we need a device capable
        // of both for our code to work.
        std::optional<uint32_t> graphicsFamily; // if device can do graphics (?)
        std::optional<uint32_t> presentFamily; // if device can draw onto screen

        bool isComplete() {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    struct SwapChainSupportDetails 
    {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    void drawFrame() {
        // Before we start the frame, we wait until previous frame is finished
        // 4th parameter indicates we wait for all fences (we have just 1 rn).
        // Final parameter is "timeout", which we set here to a very large
        // number that effectively disables the timeout.
        vkWaitForFences(device, 1, &inFlightFences[currentFrame],
            VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        uint32_t imageIndex;
        // To draw frame, need to grab an image from the swapchain. When we grab
        // an image, we signal to the semaphore that we have received an image.
        vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
            imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        updateUniformBuffer(currentFrame);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {
            imageAvailableSemaphores[currentFrame]
        };
        VkPipelineStageFlags waitStages[] = {
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
        };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        VkSemaphore signalSemaphores[] = { 
            renderFinishedSemaphores[currentFrame]
        };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        // Last parameter here (inFlightFence) is signaled once queue submission
        // is complete. This allows us to know when it is safe for the command
        // buffer to be reused (see top of function).
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, 
            inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        vkQueuePresentKHR(presentQueue, &presentInfo);

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // Creates the SWAPCHAIN image views (unfortunate name)
    void createImageViews()
    {
        swapChainImageViews.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], 
                swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    VkImageView createImageView(VkImage image, VkFormat format,
        VkImageAspectFlags aspectFlags) 
    {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        // Q: what is an array layer?
        // TMU: if you do for example VR, you need a layer per eye
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) 
            != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture image view!");
        }

        return imageView;
    }

    // Yoinked from texture chapter
    void createImage(uint32_t width, uint32_t height, VkFormat format, 
            VkImageTiling tiling, VkImageUsageFlags usage, 
            VkMemoryPropertyFlags properties, VkImage &image, 
            VkDeviceMemory &imageMemory)
    {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        // used to be VK_IMAGE_LAYOUT_UNDEFINED
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        //imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;


        if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = 
            findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) 
            != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(device, image, imageMemory, 0);
    }

    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport =
            querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat =
            chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode =
            chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        // 0 here is "special value that means that there is no maximum"
        // i.e. we are checking: is the swapchain size bounded? if so, are
        // we exceeding it by adding one more to the minimum?
        if (swapChainSupport.capabilities.maxImageCount > 0
            && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        MAX_FRAMES_IN_FLIGHT = imageCount;

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1; // #layers/image (why? see stereoscopy)
        // image usage seems important, but I don't really get it... something
        // about what operations we will use the swap chain for...
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
            | VK_IMAGE_USAGE_STORAGE_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {
            indices.graphicsFamily.value(), indices.presentFamily.value() 
        };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            // only need to set other two if sharing mode concurrent
        }

        // TODO: test flipping image with this
        createInfo.preTransform = 
            swapChainSupport.capabilities.currentTransform;

        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE; // !

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain)
            != VK_SUCCESS) {
            throw std::runtime_error("failed to create swapchain!");
        }

        // if we are here, we have a swapchain... now we need to get the images
        // out of it:

        // query how many images it will make (note: we only specified the
        // minimum no. of images in a swapChain, not maximum)
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        // move images into vector of vkImages
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, 
            swapChainImages.data());

        // saving format and extent into class members for future use
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
    {
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
            &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface,
            &formatCount, nullptr);
        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface,
                &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface,
            &presentModeCount, nullptr);
        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface,
                &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(
        const std::vector<VkSurfaceFormatKHR> &availableFormats)
    {
        for (const auto &avFormat : availableFormats) {
            if (avFormat.format == VK_FORMAT_B8G8R8A8_UNORM && 
                avFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return avFormat;
            }
        }

        // in case we don't find exactly what we want, choose first available
        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(
        const std::vector<VkPresentModeKHR> &availablePresentModes)
    {
        for (const auto &avPresentMode : availablePresentModes) {
            if (avPresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return avPresentMode;
            }
        }
        // in case mailbox (triple buffering) is unavailable, use standard fifo
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    // The resolution of swap chain images
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
    {
        if (capabilities.currentExtent.width
            != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width,
                capabilities.minImageExtent.width, 
                capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height,
                capabilities.minImageExtent.height,
                capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    void createSBT()
    {
        uint32_t groupHandleSize = rtPipelineProperties.shaderGroupHandleSize;
        groupSizeAligned = align_up(groupHandleSize,
            rtPipelineProperties.shaderGroupBaseAlignment);
        uint32_t sbtSize = groupCount * groupSizeAligned;

        std::vector<uint8_t> shaderHandleStorage(sbtSize);

        if (vkGetRayTracingShaderGroupHandlesKHR(device, rtPipeline, 0,
            groupCount, sbtSize, shaderHandleStorage.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to obtain shader group handles!");
        }

        VkMemoryAllocateFlagsInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

        createBuffer(sbtSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 
            SBT_buffer, SBT_bufferMemory, &flagsInfo);
            
        void *data;
        vkMapMemory(device, SBT_bufferMemory, 0, sbtSize, 0,
            &data);
        auto *pData = reinterpret_cast<uint8_t *>(data);

        for (uint32_t g = 0; g < groupCount; g++) {
            memcpy(pData, shaderHandleStorage.data() + 
                g * groupHandleSize, groupHandleSize);
            pData += groupSizeAligned;
        }
        vkUnmapMemory(device, SBT_bufferMemory);

    }
    /*
    void createSBT2()
    {
        const uint32_t handleSize = rtPipelineProperties.shaderGroupHandleSize;
        const uint32_t handleSizeAligned = align_up(
            rtPipelineProperties.shaderGroupHandleSize, 
            rtPipelineProperties.shaderGroupHandleAlignment);
        const uint32_t groupCount = static_cast<uint32_t>(groupCount);
        const uint32_t sbtSize = groupCount * handleSizeAligned;

        std::vector<uint8_t> shaderHandleStorage(sbtSize);
        if (vkGetRayTracingShaderGroupHandlesKHR(device, rtPipeline, 0,
            groupCount, sbtSize, shaderHandleStorage.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to obtain shader group handles!");
        }

        const VkBufferUsageFlags bufferUsageFlags = 
            VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR 
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        const VkMemoryPropertyFlags memoryUsageFlags = 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT 
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkMemoryAllocateFlagsInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

        createBuffer(handleSize, bufferUsageFlags, memoryUsageFlags, 
            rgenBuffer, rgenBufferMemory, &flagsInfo);
        createBuffer(handleSize, bufferUsageFlags, memoryUsageFlags,
            missBuffer, missBufferMemory, &flagsInfo);
        createBuffer(handleSize, bufferUsageFlags, memoryUsageFlags,
            hitBuffer, hitBufferMemory, &flagsInfo);

        // Copy handles
        raygenShaderBindingTable.map();
        missShaderBindingTable.map();
        hitShaderBindingTable.map();
        memcpy(raygenShaderBindingTable.mapped, shaderHandleStorage.data(), handleSize);
        memcpy(missShaderBindingTable.mapped, shaderHandleStorage.data() + handleSizeAligned, handleSize);
        memcpy(hitShaderBindingTable.mapped, shaderHandleStorage.data() + handleSizeAligned * 2, handleSize);
    }
    */

    uint32_t align_up(uint32_t a, uint32_t b)
    {
        uint32_t tmp = a / b;
        if ((a - tmp * b) > 0)
            return (tmp + 1) * b;
        else
            return a;
    }

    void createRayTracingPipeline()
    {
        VkPipelineLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descriptorSetLayout;
        layoutInfo.pushConstantRangeCount = 0;

        if (vkCreatePipelineLayout(device, &layoutInfo, nullptr,
            &rtPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create rt pipeline layout!");
        }

        uint32_t rgenIdx = 0;
        uint32_t hitIdx = 1;
        uint32_t missIdx = 2;

        std::array<VkPipelineShaderStageCreateInfo, shaderCount> shaderStages{};

        std::vector<char> rgenShaderCode = readFile("shaders/rgen.spv");
        std::vector<char> hitShaderCode = readFile("shaders/rchit.spv");
        std::vector<char> missShaderCode = readFile("shaders/rmiss.spv");

        VkShaderModule rgenShaderModule = createShaderModule(rgenShaderCode);
        VkShaderModule hitShaderModule = createShaderModule(hitShaderCode);
        VkShaderModule missShaderModule = createShaderModule(missShaderCode);

        shaderStages[rgenIdx].sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[rgenIdx].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        shaderStages[rgenIdx].module = rgenShaderModule;
        shaderStages[rgenIdx].pName = "main";

        shaderStages[hitIdx].sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[hitIdx].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        shaderStages[hitIdx].module = hitShaderModule;
        shaderStages[hitIdx].pName = "main";

        shaderStages[missIdx].sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[missIdx].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
        shaderStages[missIdx].module = missShaderModule;
        shaderStages[missIdx].pName = "main";

        std::array<VkRayTracingShaderGroupCreateInfoKHR, shaderCount>
            shaderGroupInfo{};

        shaderGroupInfo[rgenIdx].sType =
            VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        shaderGroupInfo[rgenIdx].type =
            VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        shaderGroupInfo[rgenIdx].generalShader = rgenIdx;
        shaderGroupInfo[rgenIdx].closestHitShader = VK_SHADER_UNUSED_KHR;
        shaderGroupInfo[rgenIdx].anyHitShader = VK_SHADER_UNUSED_KHR;
        shaderGroupInfo[rgenIdx].intersectionShader = VK_SHADER_UNUSED_KHR;
        shaderGroupInfo[rgenIdx].pNext = nullptr;
        shaderGroupInfo[rgenIdx].pShaderGroupCaptureReplayHandle = nullptr;

        shaderGroupInfo[hitIdx].sType =
            VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        shaderGroupInfo[hitIdx].type =
            VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        shaderGroupInfo[hitIdx].generalShader = VK_SHADER_UNUSED_KHR;
        shaderGroupInfo[hitIdx].closestHitShader = hitIdx;
        shaderGroupInfo[hitIdx].anyHitShader = VK_SHADER_UNUSED_KHR;
        shaderGroupInfo[hitIdx].intersectionShader = VK_SHADER_UNUSED_KHR;
        shaderGroupInfo[hitIdx].pNext = nullptr;
        shaderGroupInfo[hitIdx].pShaderGroupCaptureReplayHandle = nullptr;

        shaderGroupInfo[missIdx] = shaderGroupInfo[rgenIdx];
        shaderGroupInfo[missIdx].generalShader = missIdx;

        VkRayTracingPipelineCreateInfoKHR pipelineInfo{};
        pipelineInfo.sType = 
            VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        pipelineInfo.pNext = nullptr;
        pipelineInfo.flags = 0;
        pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.groupCount = static_cast<uint32_t>(shaderGroupInfo.size());
        pipelineInfo.pGroups = shaderGroupInfo.data();
        pipelineInfo.maxPipelineRayRecursionDepth = 1;
        pipelineInfo.layout = rtPipelineLayout;
        
        /*
        VkPipelineLibraryCreateInfoKHR libraryInfo{};
        libraryInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR;
        libraryInfo.libraryCount = 0;
        libraryInfo.pLibraries = nullptr;
        */

        /*
        pipelineInfo.pLibraryInfo = nullptr;
        pipelineInfo.pLibraryInterface = nullptr; // legal?
        pipelineInfo.basePipelineHandle = nullptr;
        pipelineInfo.basePipelineIndex = 0;
        */

		if (vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE,
            1, &pipelineInfo, nullptr, &rtPipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create rt pipeline!");
        }
    }

    void createCommandPool()
    {
        QueueFamilyIndices queueFamilyIndices = 
            findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool)
            != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    // Every unique geometric object needs to be created here
    void createGeometries()
    {
        // *   *   *   Bunny geometry (idx = 0)   *   *   *
        loadBunny(MODEL_PATH);

        // *   *   *   stupid rectangle (idx = 1)   *   *   *
        stupidRectangle();
        
        // idx 1
        loadBunny(R_MODEL_PATH);


        
        createVertexBuffer(Geometry::vertices, Geometry::vtxBuf,
            Geometry::vtxBufMem);
        createIndexBuffer(Geometry::indices, Geometry::idxBuf,
            Geometry::idxBufMem);
    }
    void createObjects()
    {
        // TODO: figure out a better way to match the indices
        createGeometries();

        objects.resize(OBJECT_COUNT);
        // *   *   *   Bunny 1   *   *   *
        objects[0].geometry = &geometries[0];
        glm::mat4 scaleBunny = glm::scale(glm::mat4(1.0f), glm::vec3(
            0.5f, 0.5f, 0.5f));
        objects[0].model_transform = 
           glm::translate(scaleBunny, glm::vec3(1.5f, 0.0f, 1.5f));
        objects[0].material.color = glm::vec3(1.0f, 1.0f, 1.0f);
        objects[0].material.reflectivity = 0.0f;
        objects[0].material.refractivity = 0.0f;
        objects[0].material.indexOfRefraction = 1.5f;
        objects[0].material.shininess = 30.0f;

        // *   *   *   floor   *   *   *
        objects[1].geometry = &geometries[1];
        glm::mat4 rectTrans = glm::scale(glm::mat4(1.0f), glm::vec3(
           30.0f, 1.0f, 30.0f));
        rectTrans = glm::rotate(0.7f*rectTrans, glm::radians(-90.0f),
           glm::vec3(1.0f, 0.0f, 0.0f));
        rectTrans = glm::translate(rectTrans, glm::vec3(0.0,0.0,-0.23));
        objects[1].model_transform = rectTrans;
        objects[1].material.color = glm::vec3(1.0f, 1.0f, 1.0f);
        objects[1].material.reflectivity = 0.4f;
        objects[1].material.refractivity = 0.0f;
        objects[1].material.indexOfRefraction = 1.45f;
        objects[1].material.shininess = 100.0f;
        
        // *   *   * WAVY  *   *   *
        objects[2].geometry = &geometries[2];
        glm::mat4  bunny2trans = glm::scale(glm::mat4(1.0f), glm::vec3(2.0f));
        bunny2trans = glm::rotate(bunny2trans, glm::radians(0.0f), glm::vec3(0.0f,1.0f,0.0f));
        bunny2trans = glm::translate(bunny2trans, glm::vec3(
            0.1f, 0.2f, -0.4f));
        objects[2].model_transform = bunny2trans;
        objects[2].material.color = glm::vec3(0.1f, 0.1f, 0.7f);
        objects[2].material.reflectivity = 0.3f;
        objects[2].material.refractivity = 1.0f;
        objects[2].material.indexOfRefraction = 1.45f;
        objects[2].material.shininess = 300.0f;

        // *   *   *   small bunny   *   *   *
        objects[3].geometry = &geometries[0];
        glm::mat4 bunny3trans = glm::rotate(0.5f*scaleBunny, glm::radians(-45.0f), glm::vec3(0.0f,1.0f,0.0f));
        bunny3trans = glm::translate(bunny3trans, glm::vec3(
            -1.2f, -0.3f, 1.2f));
        objects[3].model_transform = bunny3trans;
        objects[3].material.color = glm::vec3(1.0f, 0.0f, 0.3f);
        objects[3].material.reflectivity = 0.0f;
        objects[3].material.refractivity = 0.0f;
        objects[3].material.indexOfRefraction = 1.45f;
        objects[3].material.shininess = 100.0f;
    }

    void loadBunny(std::string MODEL_PATH)
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
            MODEL_PATH.c_str())) {
            throw std::runtime_error(warn + err);
        }

        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        // <key value pair>
        std::unordered_map<Vertex, uint32_t> uniqueVertices{};

        for (const auto &shape : shapes) {
            for (const auto &index : shape.mesh.indices) {
                Vertex vertex{};

                vertex.pos = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.normal = {
                    attrib.normals[3 * index.normal_index + 0],
                    attrib.normals[3 * index.normal_index + 1],
                    attrib.normals[3 * index.normal_index + 2],
                };

                if (uniqueVertices.count(vertex) == 0) {
                    uniqueVertices[vertex] = 
                        static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }
                indices.push_back(uniqueVertices[vertex]);
            }
        }
        Geometry bunny(vertices, indices);
        geometries.push_back(bunny);
    }

    void stupidRectangle()
    {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        glm::vec3 normal = glm::vec3(0.0f, 0.0f, 1.0f);
        Vertex v;
        v.pos = glm::vec3(-0.5f, -0.5f, 0.0f);
        v.normal = normal;
        vertices.push_back(v);
        v.pos = glm::vec3(0.5f, -0.5f, 0.0f);
        vertices.push_back(v);
        v.pos = glm::vec3(0.5f, 0.5f, 0.0f);
        vertices.push_back(v);
        v.pos = glm::vec3(-0.5f, 0.5f, 0.0f);
        vertices.push_back(v);
        
        indices = {0, 1, 2, 2, 3, 0};

        Geometry rectangle(vertices, indices);
        geometries.push_back(rectangle);
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties, VkBuffer &buffer,
        VkDeviceMemory &bufferMemory, VkMemoryAllocateFlagsInfo *flagsInfo)
    {
        // The "shape" or characteristics of our buffer
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferInfo.flags = 0; // optional

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer)
            != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.pNext = flagsInfo;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(
            memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory)
            != VK_SUCCESS) {
            throw std::runtime_error(
                "failed to allocate buffer memory!");
        }
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer,
        VkDeviceSize size)
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void createVertexBuffer(const std::vector<Vertex> &vertices, 
        VkBuffer &vertexBuffer, VkDeviceMemory &vertexBufferMemory)
    {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            // The flags we want our memory type to also have. Coherent is
            // needed to make sure we don't unmap before the mapping operation
            // completes (see tutorial).
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer, stagingBufferMemory, nullptr);

        void *data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        VkMemoryAllocateFlagsInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

        createBuffer(bufferSize,
            // What will the usage be? Here it is a vertex buffer, but also we
            // want the staging buffer to output here, hence transfer dst bit
            // (meaning buffer can be used as destination in a mem transfer
            // operation
            VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
            | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            // The optimal memory flag (why is the name "local"?)
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            vertexBuffer, vertexBufferMemory, &flagsInfo);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createIndexBuffer(const std::vector<uint32_t> &indices, 
        VkBuffer &indexBuffer, VkDeviceMemory &indexBufferMemory)
    {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer, stagingBufferMemory, nullptr);

        void *data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        VkMemoryAllocateFlagsInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

        createBuffer(bufferSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
            | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            indexBuffer, indexBufferMemory, &flagsInfo);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void createDescriptorSetLayoutRT()
    {
        std::array<VkDescriptorSetLayoutBinding, 8> bindings;

        // storage image binding
        bindings[BIND_IMG].binding = BIND_IMG;
        bindings[BIND_IMG].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[BIND_IMG].descriptorCount = 1;
        bindings[BIND_IMG].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

        bindings[BIND_TLAS].binding = BIND_TLAS;
        bindings[BIND_TLAS].descriptorType = 
            VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[BIND_TLAS].descriptorCount = 1;
        bindings[BIND_TLAS].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

        bindings[BIND_MTX].binding = BIND_MTX;
        bindings[BIND_MTX].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        bindings[BIND_MTX].descriptorCount = 1;
        bindings[BIND_MTX].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

        bindings[BIND_VERTS].binding = BIND_VERTS;
        bindings[BIND_VERTS].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[BIND_VERTS].descriptorCount = 1;
        bindings[BIND_VERTS].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR
            | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

        bindings[BIND_INDICES].binding = BIND_INDICES;
        bindings[BIND_INDICES].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[BIND_INDICES].descriptorCount = 1;
        bindings[BIND_INDICES].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR
            | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

        bindings[BIND_FIRST_VTX].binding = BIND_FIRST_VTX;
        bindings[BIND_FIRST_VTX].descriptorType = 
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[BIND_FIRST_VTX].descriptorCount = 1;
        bindings[BIND_FIRST_VTX].stageFlags = 
            VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

        bindings[BIND_PRIM_OFFSET].binding = BIND_PRIM_OFFSET;
        bindings[BIND_PRIM_OFFSET].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[BIND_PRIM_OFFSET].descriptorCount = 1;
        bindings[BIND_PRIM_OFFSET].stageFlags =
            VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

        bindings[BIND_MAT].binding = BIND_MAT;
        bindings[BIND_MAT].descriptorType =
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[BIND_MAT].descriptorCount = 1;
        bindings[BIND_MAT].stageFlags =
            VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr,
            &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create ray tracing descriptor"
                "set layout!");
        }

    }

    void createDescriptorPoolRT()
    {
        const size_t POOL_TYPES = 8;

        std::array<VkDescriptorPoolSize, POOL_TYPES> poolSizes;
        poolSizes[BIND_IMG].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSizes[BIND_IMG].descriptorCount = 
            static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        poolSizes[BIND_TLAS].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        poolSizes[BIND_TLAS].descriptorCount = 
            static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        poolSizes[BIND_MTX].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[BIND_MTX].descriptorCount =
            static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        poolSizes[BIND_VERTS].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[BIND_VERTS].descriptorCount =
            static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        poolSizes[BIND_INDICES].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[BIND_INDICES].descriptorCount =
            static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        poolSizes[BIND_FIRST_VTX].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[BIND_FIRST_VTX].descriptorCount =
            static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        poolSizes[BIND_PRIM_OFFSET].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[BIND_PRIM_OFFSET].descriptorCount =
            static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        poolSizes[BIND_MAT].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[BIND_MAT].descriptorCount =
            static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = POOL_TYPES;
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool)
            != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create rt descriptor pool!");
        }
    }

    void createDescriptorSetsRT()
    {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
            descriptorSetLayout);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount =
            static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data())
            != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        VkDescriptorBufferInfo vtxBufInfo{};
        vtxBufInfo.buffer = objects[0].geometry->vtxBuf;
        vtxBufInfo.offset = 0;
        vtxBufInfo.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo idxBufInfo{};
        idxBufInfo.buffer = objects[0].geometry->idxBuf;
        idxBufInfo.offset = 0;
        idxBufInfo.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo firstVtxInfo{};
        createFirstVerticesBuffer();
        firstVtxInfo.buffer = firstVerticesBuffer;
        firstVtxInfo.offset = 0;
        firstVtxInfo.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo primOffsetInfo{};
        createPrimOffsetsBuffer();
        primOffsetInfo.buffer = primOffsetsBuffer;
        primOffsetInfo.offset = 0;
        primOffsetInfo.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo materialInfo{};
        createMaterialsBuffer();
        materialInfo.buffer = materialsBuffer;
        materialInfo.offset = 0;
        materialInfo.range = VK_WHOLE_SIZE;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageView = swapChainImageViews[i];
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkWriteDescriptorSetAccelerationStructureKHR tlasInfo{};
            tlasInfo.sType =
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            tlasInfo.accelerationStructureCount = 1;
            tlasInfo.pAccelerationStructures = &tlas;

            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            std::array<VkWriteDescriptorSet, 8> descriptorWrite{};

            descriptorWrite[BIND_IMG].sType = 
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite[BIND_IMG].dstSet = descriptorSets[i];
            descriptorWrite[BIND_IMG].dstBinding = BIND_IMG;
            descriptorWrite[BIND_IMG].dstArrayElement = 0;
            descriptorWrite[BIND_IMG].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrite[BIND_IMG].descriptorCount = 1;
            descriptorWrite[BIND_IMG].pImageInfo = &imageInfo;

            descriptorWrite[BIND_TLAS].sType = 
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite[BIND_TLAS].dstSet = descriptorSets[i];
            descriptorWrite[BIND_TLAS].dstBinding = BIND_TLAS;
            descriptorWrite[BIND_TLAS].dstArrayElement = 0;
            descriptorWrite[BIND_TLAS].descriptorType =
                VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            descriptorWrite[BIND_TLAS].descriptorCount = 1;
            descriptorWrite[BIND_TLAS].pNext = &tlasInfo;

            descriptorWrite[BIND_MTX].sType = 
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite[BIND_MTX].dstSet = descriptorSets[i];
            descriptorWrite[BIND_MTX].dstBinding = BIND_MTX;
            descriptorWrite[BIND_MTX].dstArrayElement = 0;
            descriptorWrite[BIND_MTX].descriptorType = 
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite[BIND_MTX].descriptorCount = 1;
            descriptorWrite[BIND_MTX].pBufferInfo = &bufferInfo;

            descriptorWrite[BIND_VERTS].sType = 
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite[BIND_VERTS].dstSet = descriptorSets[i];
            descriptorWrite[BIND_VERTS].dstBinding = BIND_VERTS;
            descriptorWrite[BIND_VERTS].dstArrayElement = 0;
            descriptorWrite[BIND_VERTS].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrite[BIND_VERTS].descriptorCount = 1;
            descriptorWrite[BIND_VERTS].pBufferInfo = &vtxBufInfo;

            descriptorWrite[BIND_INDICES].sType = 
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite[BIND_INDICES].dstSet = descriptorSets[i];
            descriptorWrite[BIND_INDICES].dstBinding = BIND_INDICES;
            descriptorWrite[BIND_INDICES].dstArrayElement = 0;
            descriptorWrite[BIND_INDICES].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrite[BIND_INDICES].descriptorCount = 1;
            descriptorWrite[BIND_INDICES].pBufferInfo = &idxBufInfo;

            descriptorWrite[BIND_FIRST_VTX].sType =
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite[BIND_FIRST_VTX].dstSet = descriptorSets[i];
            descriptorWrite[BIND_FIRST_VTX].dstBinding = BIND_FIRST_VTX;
            descriptorWrite[BIND_FIRST_VTX].dstArrayElement = 0;
            descriptorWrite[BIND_FIRST_VTX].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrite[BIND_FIRST_VTX].descriptorCount = 1;
            descriptorWrite[BIND_FIRST_VTX].pBufferInfo = &firstVtxInfo;

            descriptorWrite[BIND_PRIM_OFFSET].sType =
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite[BIND_PRIM_OFFSET].dstSet = descriptorSets[i];
            descriptorWrite[BIND_PRIM_OFFSET].dstBinding = BIND_PRIM_OFFSET;
            descriptorWrite[BIND_PRIM_OFFSET].dstArrayElement = 0;
            descriptorWrite[BIND_PRIM_OFFSET].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrite[BIND_PRIM_OFFSET].descriptorCount = 1;
            descriptorWrite[BIND_PRIM_OFFSET].pBufferInfo = &primOffsetInfo;

            descriptorWrite[BIND_MAT].sType =
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite[BIND_MAT].dstSet = descriptorSets[i];
            descriptorWrite[BIND_MAT].dstBinding = BIND_MAT;
            descriptorWrite[BIND_MAT].dstArrayElement = 0;
            descriptorWrite[BIND_MAT].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrite[BIND_MAT].descriptorCount = 1;
            descriptorWrite[BIND_MAT].pBufferInfo = &materialInfo;

            /*
            descriptorWrite[BIND_MAT] = createWriteDescriptorSet(
                descriptorSets[i], BIND_MAT, &materialInfo,
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            */

            vkUpdateDescriptorSets(device, descriptorWrite.size(), 
                descriptorWrite.data(), 0, nullptr);
        }
    }

    VkWriteDescriptorSet createWriteDescriptorSet(
        VkDescriptorSet dstSet,
        uint32_t binding,
        VkDescriptorBufferInfo *bufferInfo,
        VkDescriptorType descriptorType)
    {
        VkWriteDescriptorSet d{};
        d.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        d.dstSet = dstSet;
        d.dstBinding = binding;
        d.descriptorType = descriptorType;
        d.pBufferInfo = bufferInfo;
        d.dstArrayElement = 0;
        d.descriptorCount = 1;
        return d;
    }

    void createFirstVerticesBuffer()
    {
        std::vector<uint32_t> firstVertices(objects.size());
        for (size_t i = 0; i < firstVertices.size(); i++)
            firstVertices[i] = static_cast<uint32_t>(
                objects[i].geometry->firstVtx);
        VkDeviceSize bufferSize = sizeof(uint32_t) * firstVertices.size();

        createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            firstVerticesBuffer, firstVerticesBufferMemory, nullptr);

        void *data;
        vkMapMemory(device, firstVerticesBufferMemory, 0, bufferSize,
            0, &data);
        memcpy(data, firstVertices.data(), bufferSize);
        
    }

    void createPrimOffsetsBuffer()
    {
        std::vector<uint32_t> primOffsets(objects.size());
        for (size_t i = 0; i < primOffsets.size(); i++)
            primOffsets[i] = objects[i].geometry->primOffset;
        VkDeviceSize bufferSize = sizeof(uint32_t) * primOffsets.size();

        createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            primOffsetsBuffer, primOffsetsBufferMemory, nullptr);

        void *data;
        vkMapMemory(device, primOffsetsBufferMemory, 0, bufferSize,
            0, &data);
        memcpy(data, primOffsets.data(), bufferSize);
    }

    void createMaterialsBuffer()
    {
        std::vector<Material> materials(objects.size());
        for (size_t i = 0; i < materials.size(); i++)
            materials[i] = objects[i].material;
        VkDeviceSize bufferSize = sizeof(Material) * materials.size();

        createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            materialsBuffer, materialsBufferMemory, nullptr);

        void *data;
        vkMapMemory(device, materialsBufferMemory, 0, bufferSize,
            0, &data);
        memcpy(data, materials.data(), (size_t)bufferSize);
    }

    void createUniformBuffers()
    {
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                uniformBuffers[i], uniformBuffersMemory[i], nullptr);
            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, 
                &uniformBuffersMapped[i]);
        }
    }

    void updateUniformBuffer(uint32_t currentImage)
    {
        // static function variable!
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(
            currentTime-startTime).count();

        UniformBufferObject ubo{};
        glm::vec3 rotatingPosition = 0.7f*glm::vec3(
            5.0f * glm::cos(glm::radians(time * 10.0f)),
            2.0f,
            5.0f * glm::sin(glm::radians(time * 10.0f)));

        ubo.view = glm::lookAt(rotatingPosition, glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), 
            swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1; // convert OpenGL to Vulkan (y-coord upside down)
        ubo.view_inv = glm::inverse(ubo.view);
        ubo.proj_inv = glm::inverse(ubo.proj);

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    /*
    void createDepthResources()
    {
        VkFormat depthFormat = findDepthFormat();
        createImage(swapChainExtent.width, swapChainExtent.height,
            depthFormat, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat,
            VK_IMAGE_ASPECT_DEPTH_BIT);
    }
    */

    VkFormat findSupportedFormat(const std::vector<VkFormat> &canditates,
        VkImageTiling tiling, VkFormatFeatureFlags features)
    {
        for (VkFormat format : canditates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && 
                (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    VkFormat findDepthFormat()
    {
        return findSupportedFormat({ VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    bool hasStencilComponent(VkFormat format)
    {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || 
            format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    void createBLAS(VkCommandBuffer &cmdBuffer, Geometry &geom)
    {
        VkDeviceAddress vertexBufferAddress
            = getBufferDeviceAddress(Geometry::vtxBuf);

        VkDeviceAddress indexBufferAddress
            = getBufferDeviceAddress(Geometry::idxBuf);
        
        auto vertexBindingDescription = Vertex::getBindingDescription();
        VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
        triangles.sType =
          VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = vertexBufferAddress;
        triangles.vertexStride = vertexBindingDescription.stride;
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = indexBufferAddress;
        // Unsure if it expects the whole vtx buffer or just the part we need
        triangles.maxVertex = 
            static_cast<uint32_t>(Geometry::vertices.size() - 1);
        triangles.transformData = { 0 }; // hopefully this works as before

        VkAccelerationStructureGeometryKHR geometry{};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.geometry.triangles = triangles;
        // This says we will not use any hit shaders, allowing for optimizations
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
        rangeInfo.firstVertex = geom.firstVtx;
        rangeInfo.primitiveCount = geom.primCount;
        rangeInfo.primitiveOffset = geom.primOffset;
        rangeInfo.transformOffset = 0;

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = 
          VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.flags = 
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geometry;
        // Do we build or update existing?
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
        // Specify we are doing BLAS
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
        sizeInfo.sType = 
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        // Trying to figure out what sort of parameters for size we will need
        // for building the AS
        vkGetAccelerationStructureBuildSizesKHR(device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildInfo, &rangeInfo.primitiveCount, &sizeInfo);

        VkMemoryAllocateFlagsInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

        createBuffer(sizeInfo.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            ,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            //| VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            , 
            geom.blasBuf, geom.blasBufMem, &flagsInfo);

        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = 
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.type = buildInfo.type;
        createInfo.size = sizeInfo.accelerationStructureSize;
        createInfo.buffer = geom.blasBuf;
        createInfo.offset = 0;

        if (vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, 
            &geom.blas) != VK_SUCCESS) {
            throw std::runtime_error("failed to create blas!");
        }
        
        // Finally, we set the destination AS of the build info struct
        buildInfo.dstAccelerationStructure = geom.blas;
        
        // Creating the scratch buffer
        VkBuffer scratchBuf;
        VkDeviceMemory scratchBufMem;
        createBuffer(sizeInfo.buildScratchSize,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
            /* should this be something else? */
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            ,
            scratchBuf, scratchBufMem, &flagsInfo);
        buildInfo.scratchData.deviceAddress =
            getBufferDeviceAddress(scratchBuf);

        // "Array" of rangeInfoKHR objects
        VkAccelerationStructureBuildRangeInfoKHR *pRangeInfo = &rangeInfo;
        vkCmdBuildAccelerationStructuresKHR(cmdBuffer, 1, &buildInfo, &pRangeInfo);
    }

    void createTLAS(VkCommandBuffer &cmdBuffer)
    {
        std::vector<VkAccelerationStructureInstanceKHR> instances(OBJECT_COUNT);
        for (size_t i = 0; i < OBJECT_COUNT; i++) {
            instances[i] = createBlasInstance(objects[i], i);
        }

        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
        rangeInfo.primitiveOffset = 0;
        rangeInfo.primitiveCount = OBJECT_COUNT; // Number of instances?
        rangeInfo.firstVertex = 0;
        rangeInfo.transformOffset = 0;

        VkAccelerationStructureGeometryInstancesDataKHR instancesVk{};
        instancesVk.sType = 
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        // Unsure about this. Currently, creating a buffer of device addresses, then
        // setting device address as the address of this buffer. Seems quite 
        // complicated.
        instancesVk.arrayOfPointers = VK_FALSE;
        instancesVk.data.deviceAddress = uploadInstancesBuffer(instances);
        /*
        // old solution for one instance:
        instancesVk.data.deviceAddress = getBufferDeviceAddress(
            instanceBuffer);
        */

        VkAccelerationStructureGeometryKHR geometry{};
        geometry.sType = 
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        geometry.geometry.instances = instancesVk;

        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = 
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.flags = 
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfo.geometryCount = 1; // we are building one TLAS
        buildInfo.pGeometries = &geometry;
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
        sizeInfo.sType = 
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;

        vkGetAccelerationStructureBuildSizesKHR(device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildInfo, &rangeInfo.primitiveCount, &sizeInfo);

        VkMemoryAllocateFlagsInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

        createBuffer(sizeInfo.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
            /* Should this be something else? */
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            ,
            TLAS_buffer, TLAS_bufferMemory, &flagsInfo);

        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = 
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.type = buildInfo.type;
        createInfo.size = sizeInfo.accelerationStructureSize;
        createInfo.buffer = TLAS_buffer;
        createInfo.offset = 0;

        if (vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &tlas)
            != VK_SUCCESS) {
            throw std::runtime_error("failed to create tlas!");
        }

        buildInfo.dstAccelerationStructure = tlas;

        createBuffer(sizeInfo.buildScratchSize,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            /* Should this be something else? */
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            ,
            TLAS_scratchBuffer, TLAS_scratchBufferMemory, &flagsInfo);

        buildInfo.scratchData.deviceAddress =
            getBufferDeviceAddress(TLAS_scratchBuffer);

        VkAccelerationStructureBuildRangeInfoKHR *pTlasRangeInfo 
            = &rangeInfo;
        // Hopefully, stores what it got into 'tlas'
        vkCmdBuildAccelerationStructuresKHR(cmdBuffer, 1, &buildInfo,
            &pTlasRangeInfo);
    }

    // Creates a BLAS instance, but DOESN'T UPLOAD IT!
    VkAccelerationStructureInstanceKHR createBlasInstance(
        Object &obj, uint32_t idx)
    {
        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
        addressInfo.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addressInfo.accelerationStructure = obj.geometry->blas;
        VkDeviceAddress blasAddress =
            vkGetAccelerationStructureDeviceAddressKHR(device, &addressInfo);

        VkAccelerationStructureInstanceKHR instance{};

        /*
        glm::mat4 view_proj = view * proj;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                instance.transform.matrix[i][j] = view_proj[i][j];
            }
        }
        */

        // Hope this is right
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                instance.transform.matrix[i][j] = obj.model_transform[j][i];
            }
        }

        // Might be useful in future for materials and such
        instance.instanceCustomIndex = idx;
        // for customizing which rays intersect which instances
        instance.mask = 0xFF;
        instance.instanceShaderBindingTableRecordOffset = 0;
        // Disabling backface culling
        instance.flags =
            VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = blasAddress;
        return instance;
    }

    // Uploads all instances in one buffer and returns its address
    VkDeviceAddress uploadInstancesBuffer(
        const std::vector<VkAccelerationStructureInstanceKHR> &instances)
    {
        // What is sizeof(std::vector)? Why can we rely on sizeof(vk...instance)
        // being the exact data size?
        size_t bufSize = sizeof(VkAccelerationStructureInstanceKHR)
            * instances.size();

        VkMemoryAllocateFlagsInfo flagsInfo{};
        flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

        createBuffer(bufSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            /* should this be something else? */
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            instancesBuffer, instancesBufferMemory, &flagsInfo);

        void *data;
        vkMapMemory(device, instancesBufferMemory, 0, bufSize, 0, &data);
        memcpy(data, instances.data(), bufSize);
        // probably shouldn't unmap here, but somehow in the cleanup phase
        return getBufferDeviceAddress(instancesBuffer);
        // if in pipeline, put barrier here...?
    }

    // Creates a buffer of instance device addresses, stores it on GPU, then returns
    // the address of this buffer. Needed as an input to 
    // VkAccelerationStructureGeometryInstancesDataKHR in createTLAS
    VkDeviceAddress createBlasInstanceAddressBuffer(
        std::vector<VkDeviceAddress> instanceAddresses)
    {
        VkBuffer instanceAddressBuf;
        VkDeviceMemory instanceAddressBufMem;
        createBuffer(sizeof(VkDeviceAddress) * instanceAddresses.size(),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            instanceAddressBuf, instanceAddressBufMem, 0);

        void *data;
        vkMapMemory(device, instanceAddressBufMem, 0, sizeof(VkDeviceAddress)
            * instanceAddresses.size() , 0, &data);
        memcpy(data, &instance, sizeof(instance));
        return getBufferDeviceAddress(instanceAddressBuf);
    }

    void buildAS()
    {
        blasCmdBufs.resize(GEOMETRY_COUNT);

        // Allocating BLAS cmd buffers to pool
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = blasCmdBufs.size();
        if (vkAllocateCommandBuffers(device, &allocInfo, blasCmdBufs.data())
            != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate BLAS cmd buffers!");
        }

        if (vkAllocateCommandBuffers(device, &allocInfo, &tlasCmdBuf)
            != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate TLAS cmd buffer!");
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // ! ! !
        beginInfo.pInheritanceInfo = nullptr; // for secondary command buffers

        // *   *   *   BLAS   *   *   *
        for (size_t i = 0; i < GEOMETRY_COUNT; i++) {
            if (vkBeginCommandBuffer(blasCmdBufs[i], &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error(
                    "failed to begin recording command buffer!");
            }

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            //submitInfo.pCommandBuffers = nullptr; // updated 
            submitInfo.pCommandBuffers = &blasCmdBufs[i];

            vkQueueWaitIdle(graphicsQueue);
            vkResetCommandBuffer(blasCmdBufs[i], 0);

            vkBeginCommandBuffer(blasCmdBufs[i], &beginInfo);
            createBLAS(blasCmdBufs[i], geometries[i]);
            vkEndCommandBuffer(blasCmdBufs[i]);

            vkQueueSubmit(graphicsQueue, 1, &submitInfo, nullptr);
            vkQueueWaitIdle(graphicsQueue);
            vkResetCommandBuffer(blasCmdBufs[i], 0);
        }
        //vkFreeCommandBuffers(device, commandPool, BLAS_cmdBuffers.size(), 
            //BLAS_cmdBuffers.data());

        // *   *   *   TLAS   *   *   *
        vkQueueWaitIdle(graphicsQueue);
        vkResetCommandBuffer(tlasCmdBuf, 0);
        vkBeginCommandBuffer(tlasCmdBuf, &beginInfo);
        createTLAS(tlasCmdBuf);
        vkEndCommandBuffer(tlasCmdBuf);
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &tlasCmdBuf;
        vkQueueSubmit(graphicsQueue, 1, &submitInfo, nullptr);
        vkQueueWaitIdle(graphicsQueue);
        vkResetCommandBuffer(tlasCmdBuf, 0);
        //vkFreeCommandBuffers(device, commandPool, 1, &tlasCmdBuf);
    }

    VkDeviceAddress getBufferDeviceAddress(const VkBuffer &buffer)
    {
        VkBufferDeviceAddressInfo deviceAddressInfo{};
        deviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        deviceAddressInfo.buffer = buffer;
        return vkGetBufferDeviceAddress(device, &deviceAddressInfo);
    }

    uint32_t findMemoryType(uint32_t typeFilter,
        VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            // Checking if the type of memory we want is available, and if it
            // has the flags we want (for instance, we want a memory type that
            // we can write to)
            if (typeFilter & (1 << i) 
                && (memProperties.memoryTypes[i].propertyFlags & properties) 
                == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createCommandBuffers()
    {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        // Two levels, primary and secondary. We can only submit primary buffers
        // to the queue, while secondary buffers can be called from primary
        // buffers. Thus, if we have a program that has a common use type + an
        // uncommon addition to that use, one can use secondary buffers (TMU).
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 
            static_cast<uint32_t>(commandBuffers.size());

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data())
            != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0; // ???
        beginInfo.pInheritanceInfo = nullptr; // for secondary command buffers

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error(
                "failed to begin recording command buffer!");
        }

        /*
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        // Not sure if we still need this...
        // Order of clear values should equal order of attachments
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
        clearValues[1].depthStencil = { 1.0f, 0 };
        renderPassInfo.clearValueCount = static_cast<uint32_t>(
            clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();
        */

        /*
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
            // last param configures whether we use 2ndry buffer. Here we dont.
            VK_SUBPASS_CONTENTS_INLINE);
        */

        // SBT stuff
        VkStridedDeviceAddressRegionKHR rayGenRegion{};
        rayGenRegion.deviceAddress = getBufferDeviceAddress(SBT_buffer)
            + 0 * groupSizeAligned;
        rayGenRegion.stride = groupSizeAligned;
        rayGenRegion.size = groupSizeAligned;

        VkStridedDeviceAddressRegionKHR missRegion{};
        missRegion.deviceAddress = getBufferDeviceAddress(SBT_buffer)
            + 2 * groupSizeAligned;
        missRegion.stride = groupSizeAligned;
        missRegion.size = groupSizeAligned;

        VkStridedDeviceAddressRegionKHR hitRegion{};
        hitRegion.deviceAddress = getBufferDeviceAddress(SBT_buffer)
            + 1 * groupSizeAligned;
        hitRegion.stride = groupSizeAligned;
        hitRegion.size = groupSizeAligned;
        VkStridedDeviceAddressRegionKHR callableRegion{};

        vkCmdBindPipeline(commandBuffer,
            VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline);

        // Change format to general to draw into the 2d image
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = swapChainImages[currentFrame];
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 0, NULL, 0,
            NULL, 1, &barrier);

        vkCmdBindDescriptorSets(commandBuffer,
            VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipelineLayout, 0, 1,
            &descriptorSets[currentFrame], 0, nullptr);

        vkCmdTraceRaysKHR(commandBuffer, &rayGenRegion, &missRegion,
            &hitRegion, &callableRegion, swapChainExtent.width,
            swapChainExtent.height, 1);

        // Change format to present mode for swapchain presenting
        VkImageMemoryBarrier barrier2{};
        barrier2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier2.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier2.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier2.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier2.image = swapChainImages[currentFrame];
        barrier2.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier2.subresourceRange.baseMipLevel = 0;
        barrier2.subresourceRange.levelCount = 1;
        barrier2.subresourceRange.baseArrayLayer = 0;
        barrier2.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0,
            NULL, 1, &barrier2);

        //vkCmdEndRenderPass(commandBuffer);
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // Start fence in signaled state so that we don't get stuck when drawing
        // the first image (also explained in tutorial).
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr,
                    &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr,
                    &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create semaphores/fences!");
            }
        }
    }

    void cleanupSwapChain() 
    {
        /*
        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);
        */

        /*
        for (auto framebuffer : swapChainFramebuffers) {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        */
        for (auto imageView : swapChainImageViews) {
            vkDestroyImageView(device, imageView, nullptr);
        }
        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    static std::vector<char> readFile(const std::string &filename)
    {
        // "ate" <==> "at the end", binary means read as binary
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg(); // why are we shitty casting?
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        return buffer;
    }

    VkShaderModule createShaderModule(const std::vector<char> &code)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule)
            != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }

        return shaderModule;
    }

    void createInstance() 
    {
        // testing if the validation layers we want to use are available
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not"
                "available!");
        }
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Simple Raytracer";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = 
                static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        } else {
            createInfo.enabledLayerCount = 0;
        }

        /*
        std::cout << "available extensions:\n";
        for (const auto &extension : extensions) {
            std::cout << '\t' << extension.extensionName << '\n';
        }
        */

        std::vector<const char*> extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount =
            static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }
        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
    }

    bool checkValidationLayerSupport() 
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        /*
        std::cout << "available layers: " << std::endl;
        for (const auto &layer : availableLayers)
            std::cout << layer.layerName << std::endl;
        */

        for (const char *layerName : validationLayers) {
            bool layerFound = false;

            for (const auto &layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }
        return true;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device)
    {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr,
            &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr,
            &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(),
            deviceExtensions.end());

        for (const auto &extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    std::vector<const char *> getRequiredExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char *> extensions(glfwExtensions, glfwExtensions
            + glfwExtensionCount);
        
        if (enableValidationLayers)
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        return extensions;
    }

    void createSurface() 
    {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface)
            != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    void pickPhysicalDevice() 
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan"
                "support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto &device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device; // the class member
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }

        uint32_t deviceExtensionCount;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr,
            &deviceExtensionCount, nullptr);
        std::vector<VkExtensionProperties>
            deviceExtensions(deviceExtensionCount);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr,
            &deviceExtensionCount, deviceExtensions.data());

        // std::cout << "available device extensions:\n";
        for (const auto &extension : deviceExtensions) {
            //std::cout << '\t' << extension.extensionName << '\n';
        }
    }
    
    bool isDeviceSuitable(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices = findQueueFamilies(device);
        bool extensionsSupported = checkDeviceExtensionSupport(device);
        
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport =
                querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty()
                && !swapChainSupport.presentModes.empty();
        }
        return indices.isComplete() && extensionsSupported
                && swapChainAdequate;
    }


    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) 
    {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
            nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
            queueFamilies.data());

        int i = 0;
        for (const auto &queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i,
                surface, &presentSupport);
            if (presentSupport)
                indices.presentFamily = i;

            if (indices.isComplete()) {
                break;
            }
            i++;
        }

        return indices;
    }

    void createLogicalDevice() 
    {
        // Getting rt pipeline properties (need for SBT later on)
        rtPipelineProperties.sType =
          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 deviceProperties2{};
        deviceProperties2.sType = 
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
        deviceProperties2.pNext = &rtPipelineProperties;
        vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProperties2);

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()
        };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = 
            static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.enabledExtensionCount = static_cast<uint32_t>
            (deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        // Here I'm copying from RTG II
        VkPhysicalDeviceFeatures2 features2{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2
        };
        VkPhysicalDeviceVulkan12Features features12{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
        };
        VkPhysicalDeviceVulkan11Features features11{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES 
        };
        VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR 
        };
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR 
        };

        // Ray tracing validation features
        // VkPhysicalDeviceRayTracingValidationFeaturesNV valFeatures = 
        //     { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_VALIDATION_FEATURES_NV };

        features2.pNext = &features12;
        features12.pNext = &features11;
        features11.pNext = &asFeatures;
        asFeatures.pNext = &rtPipelineFeatures;
        // rtPipelineFeatures.pNext = &valFeatures;

        vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

        // assert(valFeatures.rayTracingValidation == true);


        createInfo.pEnabledFeatures = nullptr; // dont care abt vkn 1.0 features
        createInfo.pNext = &features2;

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) 
                != VK_SUCCESS) {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, 
            &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, 
            &presentQueue);
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
        void *pUserData)
    {
        std::cerr << "validation layer: " << pCallbackData->pMessage 
            << std::endl;

        return VK_FALSE;
    }

    void setupDebugMessenger()
    {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        createInfo.sType = 
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = 
            0//VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT // might remove this
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT 
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT 
            | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = 
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT 
            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT 
            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData = nullptr;

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr,
            &debugMessenger) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to set up debug messenger!");
        }
    }

    void loadRayTracingFunctions()
    {
        vkCreateAccelerationStructureKHR = 
            (PFN_vkCreateAccelerationStructureKHR)
            vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
        if (!vkCreateAccelerationStructureKHR)
        {
            throw std::runtime_error(
                "failed to load vkCreateAccelerationStructureKHR");
        }



        vkGetAccelerationStructureBuildSizesKHR =
            (PFN_vkGetAccelerationStructureBuildSizesKHR)
            vkGetDeviceProcAddr(device, 
                "vkGetAccelerationStructureBuildSizesKHR");
        if (!vkGetAccelerationStructureBuildSizesKHR)
        {
            throw std::runtime_error(
                "failed to load vkGetAccelerationStructureBuildSizesKHR");
        }



        vkCmdBuildAccelerationStructuresKHR =
            (PFN_vkCmdBuildAccelerationStructuresKHR)
            vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
        if (!vkCmdBuildAccelerationStructuresKHR)
        {
            throw std::runtime_error(
                "failed to load vkCmdBuildAccelerationStructuresKHR");
        }



        vkGetAccelerationStructureDeviceAddressKHR =
            (PFN_vkGetAccelerationStructureDeviceAddressKHR)
            vkGetDeviceProcAddr(device, 
                "vkGetAccelerationStructureDeviceAddressKHR");
        if (!vkGetAccelerationStructureDeviceAddressKHR)
        {
            throw std::runtime_error(
                "failed to load vkGetAccelerationStructureDeviceAddressKHR");
        }



        vkGetRayTracingShaderGroupHandlesKHR =
            (PFN_vkGetRayTracingShaderGroupHandlesKHR)
            vkGetDeviceProcAddr(device,
                "vkGetRayTracingShaderGroupHandlesKHR");
        if (!vkGetRayTracingShaderGroupHandlesKHR)
        {
            throw std::runtime_error(
                "failed to load vkGetRayTracingShaderGroupHandlesKHR");
        }



        vkCmdTraceRaysKHR =
            (PFN_vkCmdTraceRaysKHR)
            vkGetDeviceProcAddr(device,
                "vkCmdTraceRaysKHR");
        if (!vkCmdTraceRaysKHR)
        {
            throw std::runtime_error(
                "failed to load vkCmdTraceRaysKHR");
        }



        vkCreateRayTracingPipelinesKHR =
            (PFN_vkCreateRayTracingPipelinesKHR)
            vkGetDeviceProcAddr(device,
                "vkCreateRayTracingPipelinesKHR");
        if (!vkCreateRayTracingPipelinesKHR)
        {
            throw std::runtime_error(
                "failed to load vkCreateRayTracingPipelinesKHR");
        }
    }
};

int main() 
{
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
