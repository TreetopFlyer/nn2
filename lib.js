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
Layer.prototype.forward = function(inCloud, inDropout){
    this.input = M.Pad(M.Clone(inCloud));
    this.output = Layer.dropout(this.deform.forward(M.Transform(this.transform, this.input)), inDropout);
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
Layer.dropout = function(inData, inChance){
    var i, j;
    for(i=0; i<inData.length; i++){
        for(j=0; j<inData[i].length; j++){
            if(Math.random() < inChance)
                inData[i][j] = 0;
        }
    }
    return inData;
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
    this.bounds = [];
    if(inMatrixArray.length > 1){
        for(var i=0; i<inMatrixArray.length-1; i++){
            this.layers[i] = new Layer(inMatrixArray[i], Layer.Activation.ReLU);
        }
    }
    this.layers[inMatrixArray.length-1] = new Layer(inMatrixArray[inMatrixArray.length-1], Layer.Activation.Sigmoid);
};

Network.prototype.forward = function(inData, inDropout){
    var observation = inData;
    for(var i=0; i<this.layers.length; i++){
        observation = this.layers[i].forward(observation, inDropout);
    }
    return observation;
};
Network.prototype.backward = function(inError, inLearningRate){
    var error = inError;
    for(var i=this.layers.length-1; i>=0; i--){
        error = this.layers[i].backward(error, inLearningRate);
    }
};

Network.prototype.train = function(inTrainingSet, inIterations, inLearningRate, inDropout, inNormalize){
    var i;
    var error, data;

    this.bounds = inTrainingSet.bounds;
    if(inNormalize)
    {
        data = M.GlobalToLocal(inTrainingSet.data, this.bounds);
    }
    else
    {
        data = inTrainingSet.data;
    }
    
    for(i=0; i<inIterations; i++){
        error = M.Subtract(this.forward(data, inDropout), inTrainingSet.labels);
        this.backward(error, inLearningRate);
    }
};
Network.prototype.error = function(inTrainingSet){
    return M.Subtract(this.forward(inTrainingSet.data, 0), inTrainingSet.labels);
};
Network.prototype.observe = function(inTrainingSet){
    return this.forward(M.GlobalToLocal(inTrainingSet.data, this.bounds), 0);
};

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

var TrainingSet = function(){
    this.data = [];
    this.labels = [];
    this.bounds = [];
};
TrainingSet.prototype.addCloud = function(inLabel, inCloud){
    var i;
    var cloudBounds;

    cloudBounds = M.Bounds(inCloud);
    this.bounds = M.Bounds(cloudBounds.concat(this.bounds));

    for(i=0; i<inCloud.length; i++){
        this.data.push(inCloud[i]);
        this.labels.push(inLabel);
    }
};
