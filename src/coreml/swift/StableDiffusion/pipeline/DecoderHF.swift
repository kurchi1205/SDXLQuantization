import PythonKit
import CoreGraphics
import CoreML

func convert_to_multi(_ latents: [MLShapedArray<Float32>]) -> [MLMultiArray] {
    let inputs: [MLMultiArray] = latents.map { sample in
            let sample_multi = sample
            return MLMultiArray(sample_multi)
        }
    return inputs
}

func convertToNumpyArray(_ latents: [MLMultiArray]) -> [PythonObject] {
    let np = Python.import("numpy")  // Import NumPy in PythonKit
    
    let inputs: [PythonObject] = latents.map { sample in
        // Convert MLMultiArray to a Swift array
        let count = sample.count
        let pointer = sample.dataPointer.bindMemory(to: Float32.self, capacity: count)
        let swiftArray = Array(UnsafeBufferPointer(start: pointer, count: count))
        
        // Convert Swift array to a NumPy array
        let numpyArray = np.array(swiftArray).reshape(sample.shape.map { Int(truncating: $0) })
        return numpyArray
    }
    return inputs
}


func loadCGImage(fromPath path: String) -> CGImage? {
    // Create a URL from the file path
    guard let url = URL(string: "file://\(path)") else {
        print("Invalid file path: \(path)")
        return nil
    }
    
    // Create an image source from the URL
    guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil) else {
        print("Failed to create image source from URL")
        return nil
    }
    
    // Create a CGImage from the image source
    let options: [CFString: Any] = [:] // Add options if needed
    guard let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, options as CFDictionary) else {
        print("Failed to create CGImage from image source")
        return nil
    }
    
    return cgImage
}

func loadAllImagesFromTmpFolder() -> [CGImage] {
    let tmpFolderPath = "/Users/prerana1298/computing/repo/SDXLQuantization/src/coreml/tmp"
    var images: [CGImage] = []
    
    do {
        // Get all files in the tmp folder
        let fileManager = FileManager.default
        let files = try fileManager.contentsOfDirectory(atPath: tmpFolderPath)

        for file in files {
            let filePath = "\(tmpFolderPath)/\(file)"
            
            // Check if the file is an image
            if file.lowercased().hasSuffix(".png") || file.lowercased().hasSuffix(".jpg") || file.lowercased().hasSuffix(".jpeg") {
                if let cgImage = loadCGImage(fromPath: filePath) {
                    images.append(cgImage)
                }
            }
        }
        
        // Delete all files in the tmp folder
        // for file in files {
        //     let filePath = "\(tmpFolderPath)/\(file)"
        //     try fileManager.removeItem(atPath: filePath)
        // }
        
    } catch {
        print("Error reading or deleting files in tmp folder: \(error)")
    }
    
    return images
}

class PyTorchModelManager {
    // private let python = Python.import("decoder_hf")
    private let python: PythonObject
    private let numpy: PythonObject
    private let torch: PythonObject
    init() {
        PythonLibrary.useVersion(3, 7)  // Specify Python version
        let sys = Python.import("sys")
        sys.path.append("/Users/prerana1298/computing/repo/SDXLQuantization/src/coreml/swift/StableDiffusion/pipeline")
        python = Python.import("decoder_hf")
        numpy = Python.import("numpy")
        torch = Python.import("torch")

    }
    
    func decodeImage(inputData: [MLShapedArray<Float32>]) -> [CGImage?] {
        let inputData_multi = convert_to_multi(inputData)
        let inputData_array = convertToNumpyArray(inputData_multi)
        // Call Python decode function
        python.decoder.decode_images(inputData_array)
        let images = loadAllImagesFromTmpFolder()
        return images
    }
}
