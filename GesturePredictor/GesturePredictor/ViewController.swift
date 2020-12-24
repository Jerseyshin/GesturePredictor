//
//  ViewController.swift
//  GesturePredictor
//
//  Created by 辛泽西 on 2020/12/24.
//

import UIKit
import CoreMotion
import CoreML

class ViewController: UIViewController {
    //MARK: Data:
    //Configuration
    static let samplesPerSecond = 25.0
    static let numberOfFeatures = 6
    static let windowSize = 20
    let modelInput: MLMultiArray! = makeMLMultiArray(numberOfSamples: windowSize)
    
    //Dislocation prediction
    static let windowOffset = 5
    static let numberOfWindows = windowSize / windowOffset
    static let bufferSize = windowSize + windowOffset * (numberOfWindows - 1)
    let dataBuffer: MLMultiArray! = makeMLMultiArray(numberOfSamples:bufferSize)
    var bufferIndex = 0
    var isDataAvailable = false
    
    //AsBytes
    static let sampleSizeAsBytes = ViewController.numberOfFeatures * MemoryLayout<Double>.stride
    static let windowOffsetAsBytes = ViewController.windowOffset * sampleSizeAsBytes
    static let windowSizeAsBytes = ViewController.windowSize * sampleSizeAsBytes
    
    //Classifier
    var predictor: GestureClassifier!
    var predictResult: GestureClassifierOutput!
    
    //Motion
    let motionManager = CMMotionManager()
    let queue = OperationQueue()
    
    
    //MARK: Properties:
    @IBOutlet weak var motionResult: UILabel!
    @IBOutlet weak var motionProb: UILabel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        do {
            predictor = try GestureClassifier(configuration: MLModelConfiguration())
        } catch {
            fatalError("Unable to use predictor!")
        }
        //motions:
        motionManager.deviceMotionUpdateInterval = 1 / ViewController.samplesPerSecond
        motionManager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: queue, withHandler: self.motionUpdateHandler)
    }

    //MARK: Functions:
    //Get sample data from sensor
    static private func makeMLMultiArray(numberOfSamples: Int) ->MLMultiArray?
    {try? MLMultiArray(
        shape: [1, numberOfSamples, numberOfFeatures] as [NSNumber],
        dataType: .double)
    }
    //Add buffer
    func addToBuffer(_ idx1: Int, _ idx2: Int, _ data: Double) {
        let index = [0, idx1, idx2] as [NSNumber]
        dataBuffer[index] = NSNumber(value: data)
    }
    //Data buffer
    func buffer(_ motionData: CMDeviceMotion) {
        for offset in [0, ViewController.windowSize] {
            let index = bufferIndex + offset
            if index >= ViewController.bufferSize {
                continue
            }
            addToBuffer(index, 0, motionData.rotationRate.x)
            addToBuffer(index, 1, motionData.rotationRate.y)
            addToBuffer(index, 2, motionData.rotationRate.z)
            addToBuffer(index, 3, motionData.userAcceleration.x)
            addToBuffer(index, 4, motionData.userAcceleration.y)
            addToBuffer(index, 5, motionData.userAcceleration.z)
        }
        bufferIndex = (bufferIndex + 1) % ViewController.windowSize
    }
    //Predict
    func predict(){
        if isDataAvailable
            && bufferIndex % ViewController.windowOffset == 0
            && bufferIndex + ViewController.windowOffset <= ViewController.windowSize {
    
            let window = bufferIndex / ViewController.windowOffset
            memcpy(
                modelInput.dataPointer,
                dataBuffer.dataPointer.advanced(by: window * ViewController.windowOffsetAsBytes),
                ViewController.windowSizeAsBytes
            )
            
            var predictorInput: GestureClassifierInput! = nil
            if predictResult != nil{
                predictorInput = GestureClassifierInput(features: modelInput, hiddenIn: predictResult.hiddenOut, cellIn: predictResult.cellOut)
            } else{
                predictorInput = GestureClassifierInput(features: modelInput)
            }
            
            do {
                predictResult = try predictor.prediction(input: predictorInput)
            } catch{
                fatalError("Unable to predict!")
            }
            
            DispatchQueue.main.async {
                let result = self.predictResult.activity
                let confidence = self.predictResult.activityProbability[result]!
                self.motionResult.text = result
                self.motionProb.text = String(format: "%.3f%%", confidence * 100)
            }
            
        }
    }
    //MotionHandler:
    func motionUpdateHandler(data motionData: CMDeviceMotion?, error: Error?){
        guard let motionData = motionData else {
            let errorText = error?.localizedDescription ?? "Unknown"
            print("Device motion update error: \(errorText)")
            return
        }
        
        buffer(motionData)
        if bufferIndex == 0 {
            isDataAvailable = true
        }
        predict()
    }
}

