#include <iostream>
#include <vector>
using namespace std;
struct Shape {
    int type; // Type of shape (e.g., 0 for circle, 1 for square, 2 for triangle)
    float x, y; // Position of the shape
    float width, height; // Size of the shape (for square)
    float radius; // Radius of circle
};
struct float3 { // Create struct for float3
    float x, y, z;
};
float3 make_float3(float x, float y, float z){
    float3 = fl;
    fl.x = x;
    fl.y = y;
    fl.z = z;
    
    return fl;
}

void preprocessImage(float* inputImage, float* outputImage, int width, int height) {
    // Implement preprocessing steps (grayscaling, resizing, e.t.c)

    // NAIVE CPU IMPLEMENTATION FROM CLASS GRAYSCALING
    // for (unsigned int ii = 0; ii < y; ii++) {
    //     for (unsigned int jj = 0; jj < x; jj++) {
    //         unsigned int idx = ii * x + jj;
    //         float r = inputImage[3 * idx];     // red value for pixel
    //         float g = inputImage[3 * idx + 1]; // green value for pixel
    //         float b = inputImage[3 * idx + 2];
    //         outputImage[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    //     }
    // }
}

int detectShapes(int width, int height, int type, Shape* detectedShapes) {
    // Implement shape detection algorithms, if any.
    // This implementation uses squares
    Shape shape;
    if type == 1{ // SQUARE
        shape.type = 1;
        shape.width = width; // Width of the square
        shape.height = height; // Height of the square
        shape.x = shape.width / 2 - 50; // Position of the square
        shape.y = shape.height / 2 - 50; 
        detectedShapes[0] = shape;
    }
    else if (type == 2) // CIRCLE
    {
        shape.type = 2;
        shape.x = width / 2;
        shape.y = height / 2;
        shape.radius = width / 4; // Radius of the circle
        detectedShapes[0] = shape;
    }
    
    return (int) sizeof(detectedShapes) / sizeof(shape);
}

void generate3DModel(const std::vector<Shape>detectedShapes, std::vector<float3>& vertices, std::vector<int>& indices) {
    // Generate 3D model from detectedShapes
    // For simplicity, this function generates a 3D cuboid based on square
    for (Shape shape : detectedShapes) {
        float width = shape.width;
        float height = shape.height;
        float3 center;
        center.x = shape.x + width / 2;
        center.y = shape.y + height / 2;
        center.z = 0.0f;

        // Generate vertices for the cuboid
        vertices.push_back(make_float3(center.x - width / 2, center.y - height / 2, 0.0f)); // Bottom front left
        vertices.push_back(make_float3(center.x + width / 2, center.y - height / 2, 0.0f)); // Bottom front right
        vertices.push_back(make_float3(center.x + width / 2, center.y + height / 2, 0.0f)); // Bottom back right
        vertices.push_back(make_float3(center.x - width / 2, center.y + height / 2, 0.0f)); // Bottom back left
        vertices.push_back(make_float3(center.x - width / 2, center.y - height / 2, height)); // Top front left
        vertices.push_back(make_float3(center.x + width / 2, center.y - height / 2, height)); // Top front right
        vertices.push_back(make_float3(center.x + width / 2, center.y + height / 2, height)); // Top back right
        vertices.push_back(make_float3(center.x - width / 2, center.y + height / 2, height)); // Top back left

        // Generate indices for the cuboid
        int baseIdx = vertices.size() - 8;
        indices.insert(indices.end(), {
            baseIdx, baseIdx + 1, baseIdx + 2, baseIdx,
            baseIdx + 2, baseIdx + 3, baseIdx + 4, baseIdx + 5,
            baseIdx + 6, baseIdx + 4, baseIdx + 7, baseIdx,
            baseIdx + 7, baseIdx + 3, baseIdx + 1, baseIdx,
            baseIdx + 5, baseIdx + 6, baseIdx + 2, baseIdx + 5,
            baseIdx + 4, baseIdx, baseIdx + 1, baseIdx + 5,
            baseIdx + 1, baseIdx + 2, baseIdx + 6, baseIdx + 1,
            baseIdx + 6, baseIdx + 2, baseIdx + 3, baseIdx + 7,
            baseIdx + 6, baseIdx + 3, baseIdx + 6, baseIdx + 7
        });
    }
}

void generate3DModel(const std::vector<Shape>& detectedShapes, std::vector<float3>& vertices, std::vector<int>& indices) {
    // Render 3D image

    // utilize input and output
}