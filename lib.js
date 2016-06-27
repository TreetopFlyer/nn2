/*
M.:
Transform
Transpose
Multiply
Subtract
Scale
Pad/Unpad
Sigmoid
Derivative
*/

var Layer = function(inTransformer, inDeformer){
    this.transform = inTransformer;
    this.deform = inDeformer;
    this.input = [];
    this.output = [];
};
Layer.prototype.forward = function(inCloud){
    this.input = M.Pad(inCloud);
    this.output = this.deform.forward( M.Transform(this.transform, this.input) );
    return this.output;
};
Layer.prototype.backward = function(inErrors, inLearningRate){
    var transpose = M.Transpose(this.transform);
    var errorDerivative = M.Multiply(inErrors, this.deform.backward(this.output));
    var delta;
    var i;

    // subtract the outer product of input and errorDerivative from the matrix (scaled by the learning rate)
    for(i=0; i<errorDerivative.length; i++){
        delta = M.Outer(this.input[i], errorDerivative[i]);
        this.transform = M.Subtract(this.transform, M.Scale(delta, inLearningRate));
    }
    return M.Unpad(M.Transform( transpose, errorDerivative));
};
Layer.Activation = {
    Sigmoid:{
        forward : function(inData){return M.Sigmoid(inData);},
        backward : function(inData){return M.Derivative(inData);}
    },
    ReLU:{
        forward : function(inData){
            var i, j;
            var matrix, vector;
            matrix = [];
            for(i=0; i<inData.length; i++){
                vector =[];
                for(j=0; j<inData[i].length; j++){
                    if(inData[i][j] <= 0){
                        vector[j] = 0;
                    }
                    else{
                        vector[j] = inData[i][j];
                    }
                }
                matrix[i] = vector;
            }
            return matrix;
        },
        backward : function(inData){
            var i, j;
            var matrix, vector;
            matrix = [];
            for(i=0; i<inData.length; i++){
                vector =[];
                for(j=0; j<inData[i].length; j++){
                    if(inData[i][j] <= 0){
                        vector[j] = 0.01;
                    }
                    else{
                        vector[j] = 1;
                    }
                }
                matrix[i] = vector;
            }
            return matrix;
        }
    }
};

//the members of inMatrixArray have to be padded before they are passed in
var Network = function(inMatrixArray){
    this.layers = [];
    if(inMatrixArray.length > 1){
        for(var i=0; i<inMatrixArray.length-1; i++){
            this.layers[i] = new Layer(inMatrixArray[i], Layer.Activation.ReLU);
        }
    }
    this.layers[inMatrixArray.length-1] = new Layer(inMatrixArray[inMatrixArray.length-1], Layer.Activation.Sigmoid);
};
Network.prototype.observe = function(inData){
    var observation = inData;
    for(var i=0; i<this.layers.length; i++){
        observation = this.layers[i].forward(observation);
    }
    return observation;
};
Network.prototype.backpropogate = function(inError, inLearningRate){
    var error = inError;
    for(var i=this.layers.length-1; i>=0; i--){
        error = this.layers[i].backward(error, inLearningRate);
    }
};
Network.prototype.train = function(inData, inLabels, inIterations, inLearningRate){
    var i, j;
    var error;
    var sum;
    for(i=0; i<inIterations; i++){
        error = M.Subtract(this.observe(inData), inLabels);
        this.backpropogate(error, inLearningRate);
    }
    return error;
}
Network.generateMatricies = function(inShapeArray){
    var output = [];
    var currentMatrix;
    var currentVector;
    var i, j, k;
    for(i=0; i<inShapeArray.length-1; i++){
        //inShapeArray[i]   = number of inputs : number of components in each vector
        //inShapeArray[i+1] = number of outputs: number of vectors in the matrix
        currentMatrix = [];
        for(j=0; j<inShapeArray[i+1]; j++){
            currentVector = [];
            // padding occurs here! vectors are padded with an extra component
            for(k=0; k<inShapeArray[i]+1; k++){
                currentVector[k] = Math.random() - 0.5;
            }
            currentMatrix[j] = currentVector;
        }
        output[i] = currentMatrix;
    }
    return output;
};
