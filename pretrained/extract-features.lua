--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  extracts features from an image using a trained model
--

-- USAGE
-- SINGLE FILE MODE
--          th extract-features.lua [MODEL] [FILE] ...
--
-- BATCH MODE
--          th extract-features.lua [MODEL] [BATCH_SIZE] [DIRECTORY_CONTAINING_IMAGES]
--


require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'
local t = require '../datasets/transforms'


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 ResNet extracting features script')
cmd:text()
cmd:text('Options:')
cmd:option('-model',       '', 'Path to the model')
cmd:option('-data',        '', 'Path to the data file or folder')
cmd:option('-batchSize',   1, 'Batch size')
cmd:option('-tenCrop',     'false', 'Ten-crop testing')
cmd:option('-recursive',   'false', 'Recursive folder lookup')
cmd:option('-layer',       -2, 'From which layer to extract features. Negative numbers count from the end. Default is -2 (before FC layer).')
cmd:text()

local opt = cmd:parse(arg or {})
opt.tenCrop = opt.tenCrop ~= 'false'
opt.recursive = opt.recursive ~= 'false'


-- get the list of files
local list_of_filenames = {}

if not paths.filep(opt.model) then
    io.stderr:write('Model file not found at ' .. f .. '\n')
    os.exit(1)
end


if paths.dirp(opt.data) then -- batch mode ; collect file from directory

    local lfs  = require 'lfs'

    for file in lfs.dir(opt.data) do -- get the list of the files
        if file~="." and file~=".." then
            table.insert(list_of_filenames, opt.data..'/'..file)
        end
    end

else -- single file mode ; collect file from command line
    -- TODO: how to do it with opts??
    -- for i=2, #arg do
    --     f = arg[i]
    --     if not paths.filep(f) then
    --       io.stderr:write('file not found: ' .. f .. '\n')
    --       os.exit(1)
    --     else
    --        table.insert(list_of_filenames, f)
    --     end
    -- end
end

local number_of_files = #list_of_filenames

if opt.batchSize > number_of_files then opt.batchSize = number_of_files end

-- Load the model
local model = torch.load(opt.model):cuda()

local layer_to_extract = opt.layer
if layer_to_extract < 0 then
   layer_to_extract = #model.modules - layer_to_extract + 1
end

-- -- Remove the fully connected layer
-- assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
-- model:remove(#model.modules)

-- Evaluate mode
model:evaluate()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.Scale(256),
   t.ColorNormalize(meanstd),
   t.CenterCrop(224),
}

local features

for i=1,number_of_files,opt.batchSize do
    local img_batch = torch.FloatTensor(opt.batchSize, 3, 224, 224) -- batch numbers are the 3 channels and size of transform

    -- preprocess the images for the batch
    local image_count = 0
    for j=1,opt.batchSize do
        img_name = list_of_filenames[i+j-1]

        if img_name  ~= nil then
            image_count = image_count + 1
            local img = image.load(img_name, 3, 'float')
            img = transform(img)
            img_batch[{j, {}, {}, {} }] = img
        end
    end

    -- if this is last batch it may not be the same size, so check that
    if image_count ~= opt.batchSize then
        img_batch = img_batch[{{1,image_count}, {}, {}, {} } ]
    end

   model:forward(img_batch:cuda())
   local output = model.modules[layer_to_extract].output


   -- this is necesary because the model outputs different dimension based on size of input
   if output:nDimension() == 1 then output = torch.reshape(output, 1, output:size(1)) end

   if not features then
       features = torch.FloatTensor(number_of_files, output:size(2)):zero()
   end
   features[{ {i, i-1+image_count}, {}  } ]:copy(output)

end

torch.save('features.t7', {features=features, image_list=list_of_filenames})
print('saved features to features.t7')
