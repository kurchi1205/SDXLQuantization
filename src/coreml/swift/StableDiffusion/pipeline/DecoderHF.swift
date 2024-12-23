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

func convert_to_array(_ latents: [MLMultiArray]) -> [Array] {
    let inputs: [Array] = latents.map { sample in
            let sample_array = sample
            return Array(UnsafeBufferPointer(start: multiArray.dataPointer.assumingMemoryBound(to: Float32.self), count: multiArray.count))
        }
    return inputs
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
    
    func decodeImage(inputData: [MLShapedArray<Float32>]) -> [MLShapedArray<Float32>] {
        
        // // Convert MLShapedArray to numpy array
        // let shape = inputData.shape
        // let buffer = numpy.array(Array(UnsafeBufferPointer(start: inputData.scalars, count: inputData.count)))
        // let reshaped = buffer.reshape(shape.map { Int($0) })
        let inputData_multi = convert_to_multi(inputData)
        print(inputData_multi)
        let inputData_array = convert_to_array(inputData_multi)
        print(inputData_array)
        // Call Python decode function
        // let result = python.decoder.decode_images(reshaped)
        
        // Convert result back to MLShapedArray
        // let resultArray = Array<Float32>(numpy: result.numpy())
        return inputData
        
    }
}
