require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'image'
require 'csvigo'

train_path_prefix = 'Final_Training/Images/'
test_path_prefix = 'Final_Test/Images/'

-- Read Training CSV and Images
train_csv = { Filename = {}, Width = {}, Height = {}, X1 = {}, Y1 = {}, X2 = {}, Y2 = {}, Label = {} }
trainData = { data = {}, labels = {}, size = function() return #trainData.data end }

for i = 0, 42 do
	-- Get the path to the file	
	local pathToFolder = train_path_prefix .. string.format( "%05d/", i )
	local pathToCSV = pathToFolder .. string.format( "GT-%05d.csv", i )

	-- Read the CSV
	local csvData = csvigo.load{ path = pathToCSV, separator = ';', mode = 'tidy', header = true }

	-- Loop through the data
	for i = 1, #(csvData.Filename) do
		table.insert( train_csv.Filename, pathToFolder .. csvData.Filename[i] )
		table.insert( train_csv.Width, tonumber( csvData.Width[i] ) )
		table.insert( train_csv.Height, tonumber( csvData.Height[i] ) )
		table.insert( train_csv.X1, tonumber( csvData["Roi.X1"][i] ) )
		table.insert( train_csv.Y1, tonumber( csvData["Roi.Y1"][i] ) )
		table.insert( train_csv.X2, tonumber( csvData["Roi.X2"][i] ) )
		table.insert( train_csv.Y2, tonumber( csvData["Roi.Y2"][i] ) )
		table.insert( train_csv.Label, tonumber(csvData["ClassId\r"][i]) + 1 )
	end
end

for i, v in ipairs(train_csv.Filename) do
	-- Load the image
	local img = image.load(v)

	-- Crop and scale
	local cropImg = image.crop( img, train_csv.X1[i], train_csv.Y1[i], train_csv.X2[i], train_csv.Y2[i] )
	local scaleImg = image.scale( cropImg, 32, 32 )

	-- Insert new image and label
	table.insert( trainData.data, scaleImg )
	table.insert( trainData.labels, train_csv.Label[i] )
end

-- Read Testing CSV and Images
test_csv = { Filename = {}, Width = {}, Height = {}, X1 = {}, Y1 = {}, X2 = {}, Y2 = {}, Label = {} }
testData = { data = {}, labels = {}, size = function() return #testData.data end }

do
	-- Get the path to the file	
	local pathToCSV = test_path_prefix .. "GT-final_test.csv"

	-- Read the CSV
	local csvData = csvigo.load{ path = pathToCSV, separator = ';', mode = 'tidy', header = true }

	-- Loop through the data
	for i = 1, #(csvData.Filename) do
		table.insert( test_csv.Filename, test_path_prefix .. csvData.Filename[i] )
		table.insert( test_csv.Width, csvData.Width[i] )
		table.insert( test_csv.Height, csvData.Height[i] )
		table.insert( test_csv.X1, csvData["Roi.X1"][i] )
		table.insert( test_csv.Y1, csvData["Roi.Y1"][i] )
		table.insert( test_csv.X2, csvData["Roi.X2"][i] )
		table.insert( test_csv.Y2, csvData["Roi.Y2"][i] )
		table.insert( test_csv.Label, tonumber(csvData["ClassId\r"][i]) + 1 )
	end
end

for i, v in ipairs(test_csv.Filename) do
	-- Load the image
	local img = image.load(v)

	-- Crop and scale
	local cropImg = image.crop( img, test_csv.X1[i], test_csv.Y1[i], test_csv.X2[i], test_csv.Y2[i] )
	local scaleImg = image.scale( cropImg, 32, 32 )

	-- Insert new image and label
	table.insert( testData.data, scaleImg )
	table.insert( testData.labels, test_csv.Label[i] )
end

-- Nulliy CSV data
train_csv = nil
test_csv = nil
collectgarbage()

function train_model(lrn_rate, wght_decay, momntm, lrn_rate_decay)
    epoch = epoch or 1
    local time = sys.clock()
    shuffle = torch.randperm(trainData:size())
    batchSize = 64
    for t = 1, trainData:size(), batchSize do
        local inputs = {}
        local targets = {}
        
        for i = t, math.min(t + batchSize - 1, trainData:size()) do
            local input = trainData.data[shuffle[i]]:double()
            local target = trainData.labels[shuffle[i]]
            
            table.insert(inputs, input)
            table.insert(targets, target)
        end
        
        local feval = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end
            
            gradParameters:zero() --Why?
            
            local f = 0

            for i = 1, #inputs do
                local output = model:forward(inputs[i])
                output = torch.reshape(output, 43)
                local err = criterion:forward(output, targets[i])
                f = f + err -- f is the sum of the errors for the inputs
                local df_do = criterion:backward(output:double(), targets[i])
                model:backward(inputs[i], df_do)
                confusion:add(output, targets[i])
            end
            
            gradParameters:div(#inputs)
            f = f / #inputs
            
            return f, gradParameters
        end
        
        config = {learningRate = lrn_rate, weightDecay = wght_decay, 
            momentum = momntm, learningRateDecay = lrn_rate_decay}
        optim.sgd(feval, parameters, config)
    end
    
    print(confusion)
    confusion:zero()
        
    local filename = paths.concat('model.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    torch.save(filename, model)
        
    epoch = epoch + 1
    
end

-- Model 
nstates = {32, 64}
fanin = {1,4}
filtsize = 5
poolsize = 2
nhidden = 256
ninput = 32*32
noutput = 43
kernel = image.gaussian1D(7)

reshaper = nn.Reshape(ninput)

model = nn.Sequential()
model:add(nn.SpatialContrastiveNormalization(3, kernel))

model:add(nn.SpatialConvolution(3, nstates[1], filtsize, filtsize))
model:add(nn.PReLU())
model:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

model:add(nn.SpatialConvolution(nstates[1], nstates[2], 5, 5))
model:add(nn.PReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))


model:add(nn.Reshape(nstates[2]*5*5))

model:add(nn.Linear(nstates[2]*5*5,nhidden))
model:add(nn.PReLU())
model:add(nn.Linear(nhidden, noutput))
model:add(nn.PReLU())
model:add(nn.LogSoftMax())

print(model)

criterion = nn.ClassNLLCriterion()

classes = {}

for i = 1, 43 do
	table.insert( classes, i )
end

confusion = optim.ConfusionMatrix(classes)


collectgarbage()

num_training_loops = 15
lrn_rate = 0.003
wght_decay = 0.01
momntm = 0.01
lrn_rate_decay = 5e-7

if model then
    parameters, gradParameters = model: getParameters()
end

for i = 1,num_training_loops do
        print("________", i, "___________")
        train_model(lrn_rate, wght_decay, momntm, lrn_rate_decay)
end

test_confusion = optim.ConfusionMatrix(classes)

function test_model()
    for t = 1, testData:size() do
        local input = testData.data[t]:double()
        local target = testData.labels[t]

        local pred = model:forward(input)
        pred = torch.reshape(pred, 43)
        test_confusion:add(pred, target)
    end
end

test_model()
print(test_confusion)