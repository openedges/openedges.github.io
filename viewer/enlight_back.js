/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// var enlight = enlight || {};
// var base = base || require('./base');
// var flatbuffers = flatbuffers || require('flatbuffers').flatbuffers;
// var long = long || { Lo

import * as enlight_schema from './enlight-schema.js';
import * as flatbuffers from './flatbuffers-custom.js';

const enlight = {};

enlight.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const match_result = identifier.endsWith('enlight');
        console.log(`match_result: ${match_result}`);
        if (match_result) {
            context.type = 'enlight';
            context.target = 'enlight';
        }
    }

    async open(context) {
        console.log('open');
        console.log('context.stream = ', context.stream);
        const identifier = context.identifier;
        let model = null;
        try {
            const buffer = context.stream._buffer;
            const byteBuffer = new flatbuffers.ByteBuffer(buffer);
            enlight.schema = enlight_schema.Enlight_Schema;
            const test = new enlight_schema.Network();
            model = test.getRootAsNetwork(byteBuffer);
            console.log(`[enlight.ModelFactory]getRootAsNetwork: ${model.layersLength()}`);
        } catch (error) {
            let message = error && error.message ? error.message : error.toString();
            message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
            console.log(`${message} in '${identifier}'.`);
            throw new enlight.Error(`${message} in '${identifier}'.`);
        }

        const metadata = await enlight.Metadata.open(context);
        console.log(`[enlight.ModelFactory]metadata: ${metadata}`);
        try {
            return new enlight.Model(model, metadata);
        } catch (error) {
            let message = error && error.message ? error.message : error.toString();
            message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
            console.log(`${message} in '${identifier}'.`);
            throw new new enlight.Error(`${message} in '${identifier}'.`);
        }
    }
};

enlight.Model = class {

    constructor(model, metadata) {
        this._graphs = [];
        this._graphs.push(new enlight.Graph(model, metadata));
        this._net_info = new enlight.NetInfo(model.netinfo());
    }

    get format() {
        return 'EnlightNN';
    }

    get model() {
        return this._net_info.model_name;
    }

    get model_type() {
        return this._net_info.model_type;
    }

    get has_detection_layer() {
        return this._net_info.has_detection_layer;
    }

    get num_class() {
        return this._net_info.num_class;
    }

    get class_labels() {
        const labels = this._net_info.class_labels;

        let new_labels = [];

        for (let i = 0 ; i < labels.length ; i++) {
            const new_label = `#${i} : ${labels[i]}`;
            new_labels.push(new_label);
        }

        return new_labels.join('\n');
    }

    get has_score() {
        return this._net_info.has_score;
    }

    get mAP() {
        return this._net_info.mAP;
    }

    get top5() {
        return this._net_info.top5;
    }

    get top1() {
        return this._net_info.top1;
    }

    get evaluation_dataset() {
        return this._net_info.evaluation_dataset;
    }

    get is_fused_normalization() {
        return this._net_info.is_fused_normalization;
    }

    get norm_mean() {
        return this._net_info.norm_mean;
    }

    get norm_std() {
        return this._net_info.norm_std;
    }

    get optimizations() {
        return this._net_info.optimization;
    }

    get is_tracked() {
        return this._net_info.is_tracked;
    }

    get has_histogram() {
        return this._net_info.has_histogram;
    }

    get track_dataset() {
        return this._net_info.track_dataset;
    }

    get num_images() {
        return this._net_info.num_images;
    }

    get is_quantized() {
        return this._net_info.is_quantized;
    }

    get quantization_method() {
        return this._net_info.quantization_method;
    }

    get m_std_8() {
        return this._net_info.m_std_8;
    }

    get m_std_4() {
        return this._net_info.m_std_4;
    }

    get m_std_ratio() {
        return this._net_info.m_std_ratio;
    }

    get clip_min_max() {
        return this._net_info.clip_min_max;
    }

    get iter_weight_mean_correction() {
        return this._net_info.iter_weight_mean_correction;
    }

    get quantize_post_process() {
        return this._net_info.quantize_post_process;
    }

    get graphs() {
        return this._graphs;
    }

    get sdk_version() {
        return this._net_info.sdk_version;
    }
};

enlight.NetInfo = class {
    constructor(info) {
        this._model_name = info.model();
        this._model_type = info.type();
        this._has_detection_layer =  info.hasDetectionLayer();
        this._num_class = info.numClass();

        this._class_labels = [];
        for (let i = 0; i < info.classLabelsLength(); i++) {
            this._class_labels.push(info.classLabels(i));
        }

        this._has_score = info.hasScore();
        this._mAP = info.mAP();
        this._top5 = info.top5();
        this._top1 = info.top1();
        this._evaluation_dataset = info.evaluationDataset();

        this._is_fused_normalization = info.isFusedNormalization();
        this._norm_mean = info.normMeanArray();
        this._norm_std = info.normStdArray();

        this._optimization = [];
        for (let i = 0; i < info.optimizationLength(); i++) {
            this._optimization.push(info.optimization(i));
        }
        this._is_tracked = info.isTracked();
        this._has_histogram = info.hasHistogram();
        this._track_dataset = info.trackDataset();
        this._num_images = info.numImages();

        this._is_quantized = info.isQuantized();
        this._quantization_method = info.quantizationMethod();
        this._m_std_8 = info.mStd8();
        this._m_std_4 = info.mStd4();
        this._m_std_ratio = info.mStdRatio();
        this._clip_min_max = info.clipMinMax();
        this._iter_weight_mean_correction = info.iterWeightMeanCorrection();
        this._quantize_post_process = info.quantizePostProcess();

        this._sdk_version = info.sdkVersion();
    }

    get model_name() {
        return this._model_name;
    }

    get model_type() {
        return this._model_type;
    }

    get has_detection_layer() {
        return this._has_detection_layer;
    }

    get num_class() {
        return this._num_class;
    }

    get class_labels() {
        return this._class_labels;
    }

    get has_score() {
        return this._has_score;
    }

    get mAP() {
        return this._mAP;
    }

    get top5() {
        return this._top5;
    }

    get top1() {
        return this._top1;
    }

    get evaluation_dataset() {
        return this._evaluation_dataset;
    }

    get is_fused_normalization() {
        return this._is_fused_normalization;
    }

    get norm_mean() {
        return this._norm_mean;
    }

    get norm_std() {
        return this._norm_std;
    }

    get optimization() {
        return this._optimization;
    }

    get is_tracked() {
        return this._is_tracked;
    }

    get has_histogram() {
        return this._has_histogram;
    }

    get track_dataset() {
        return this._track_dataset;
    }

    get num_images() {
        return this._num_images;
    }

    get is_quantized() {
        return this._is_quantized;
    }

    get quantization_method() {
        return this._quantization_method;
    }

    get m_std_8() {
        return this._m_std_8;
    }

    get m_std_4() {
        return this._m_std_4;
    }

    get m_std_ratio() {
        return this._m_std_ratio;
    }

    get clip_min_max() {
        return this._clip_min_max;
    }

    get iter_weight_mean_correction() {
        return this._iter_weight_mean_correction;
    }

    get quantize_post_process() {
        return this._quantize_post_process;
    }

    get sdk_version() {
        return this._sdk_version;
    }
};

enlight.Graph = class {

    constructor(graph, metadata) {
        this._name = '';
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];

        let params = {};

        // generate parameters
        let paramIdx = 0;
        for (let j = 0; j < graph.layersLength(); j++) {
            let base = enlight.Node.getBase(graph.layers(j));
            for (let i = 0 ; i < base.outputSlotsLength() ; i++) {
                let slot = base.outputSlots(i);
                let key = enlight.Parameter.makeKey(base.index(), i);
                let name = paramIdx.toString();

                let stats = null;
                let threshold = null;

                if (slot.statisticsEnabled()) {
                    stats = [slot.min(), slot.max(), slot.mean(), slot.std()];
                }

                if (slot.thresholdEnabled()) {
                    threshold = slot.threshold();
                }

                let args = [new enlight.Argument(name, slot.tensorInfo(), null, stats, threshold)];
                params[key] = new enlight.Parameter(name, name, args);
                paramIdx++;
            }
        }

        // generate nodes
        for (let j = 0; j < graph.layersLength(); j++) {
            this._nodes.push(new enlight.Node(graph.layers(j), params, metadata, false));
        }

        // link inputs
        for (let k = 0; k < graph.inputIdsLength(); k++) {
            // need to do something?
        }

        // link outputs
        for (let l = 0; l < graph.outputIdsLength(); l++) {
            // need to do something?
        }
    }

    get name() {
        return this._name;
    }

    get groups() {
        return true;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }
};

enlight.Node = class {

    constructor(layer, params, metadata, fused) {
        this.metadata = metadata;
        const op_type = enlight.schema.LayerName[layer.layerType()].replace(/Layer$/, '');

        // this.name = '';
        const outputs = [];
        const inputs = [];
        this._category = '';
        this._group_idc = null;
        this._position_in_group = -1;

        let base = null;

        if (!fused) {
            base = enlight.Node.getBase(layer);
        }

        this.type = {};
        this.type.name = op_type;

        let name = '';
        const chain = [];
        if (base) {
            name = base.layerName();
            this._group_idc = layer.groupIdc();
            this._position_in_group = layer.positionInGroup();

            for (let i = 0; i < base.inputSlotsLength(); i++) {
                const srcConnection = base.inputSlots(i).connection();
                const srcLayerIdx = srcConnection.sourceLayerIndex();
                const srcOutputIdx = srcConnection.outputSlotIndex();

                inputs.push(params[enlight.Parameter.makeKey(srcLayerIdx, srcOutputIdx)]);
            }

            for (let j = 0; j < base.outputSlotsLength(); j++) {
                outputs.push(params[enlight.Parameter.makeKey(base.index(), j)]);
            }

            for (let j = 0; j < layer.fusedLayersLength(); j++) {
                const fusedLayer = layer.fusedLayers(j);
                chain.push(new enlight.Node(fusedLayer, params, metadata, true));
            }
        }

        this.inputs = inputs || [];
        this.outputs = outputs || [];
        this.name = name || '';
        this.chain = chain || [];
        this.attributes = this.setAttribute(layer, fused);
    }

    get operator() {
        return this.type;
    }

    // get name() {
    //     return this.name;
    // }

    get domain() {
        return null;
    }

    get documentation() {
        return '';
    }

    get group() {
        return this._group_idc;
    }

    get category() {
        return this._category;
    }

    // get inputs() {
    //     return this.inputs;
    // }

    // get outputs() {
    //     return this.outputs;
    // }

    // get chain() {
    //     return this.chain;
    // }

    // get attributes() {
    //     return this.attributes;
    // }

    get group_idc() {
        return this._group_idc;
    }

    get position_in_group() {
        return this._position_in_group;
    }

    static castLayer(layer) {
        let layerType = layer.layerType();

        for (let k of Object.keys(enlight.schema.Layer)) {
            if (layerType == enlight.schema.Layer[k]) 
                return layer.layer(new enlight.schema[k]);
        }
        return null;
    }

    static getBase(layer) {
        return layer.base();
    }

    getDescriptor(layer) {
        if (layer == null)
            return null;

        return layer.descriptor();
    }

    getAttr(descriptor, key) {
        if (typeof descriptor[key] == "undefined")
            return "undefined";

        if (typeof descriptor[key + "Length"] != "undefined") {
            let values = [];

            for (let i = 0 ; i < descriptor[key + "Length"]() ; i++)
                values.push(descriptor[key](i));

            return values.join(", ");
        }
        else {
            return descriptor[key]();
        }
    }

    getAttrOptionKeys(schema) {
        if(typeof schema["attributes_option_keys"] != "undefined")
            return schema.attributes_option_keys;
        
        return null;
    }

    getAttrOptionFlag(layer, schema) {
        let keys = this.getAttrOptionKeys(schema);
        if(!keys)
            return false;

        let descriptor = this.getDescriptor(layer);
        if(!descriptor)
            return false;

        for(let i = 0 ; i < keys.length; i++) {
            let key = keys[i].src;
            let flag = this.getAttr(descriptor, key);
            if(!flag)
                return false;
        }

        return true;
    }
            
    packAttr(layer, attr) {
        let descriptor = this.getDescriptor(layer);

        let key  = attr.src;
        let type = attr.src_type;

        if (typeof type != "undefined") {
            let value = this.getAttr(descriptor, key);
            if (typeof enlight.schema[type + "Name"] != "undefined")
                return enlight.schema[type + "Name"][value];
            else
                return value;
        }
        else if (Array.isArray(key)) {
            let values = [];
            for (let i = 0 ; i < key.length ; i++) {
                values.push(this.getAttr(descriptor, key[i]));
            }
            return values.join(", ");
        }
        else {
            let values = this.getAttr(descriptor, key)

            if (Array.isArray(values))
                return values.join(", ");
            else
                return values;
        }
    }

    getRowPartitionInfo(layer) {
        let descriptor = this.getDescriptor(layer)

        let partitionDescriptor = descriptor['rowPartitionData']()

        if (partitionDescriptor == null) {
            return false
        }

        let partitionData = [];
        let start;
        let end;
        let info;

        let partitionDataLength = partitionDescriptor['rowPartitionDataLength']();
        for (let i = 0; i < partitionDataLength; i++) {
            let d = partitionDescriptor['rowPartitionData'](i);

            start = d['start']();
            end = d['end']();

            info = '[' + start.toString() + ' ,' + end.toString() + ')';

            partitionData.push(info);
        }

        return partitionData;
    }

    getChannelPartitionInfo(layer) {
        let descriptor = this.getDescriptor(layer)

        let partitionDescriptor = descriptor['channelPartitionData']()

        if (partitionDescriptor == null) {
            return false
        }

        let partitionPos = [];
        let partitionData = [];

        let start;
        let end;
        let positionInGroup;

        let partitionDataLength = partitionDescriptor['channelPartitionDataLength']();

        for (let i = 0; i < partitionDataLength; i++) {
            let d = partitionDescriptor['channelPartitionData'](i);

            positionInGroup = d['positionInGroup']();

            let channelSizeLength = d['partitionSizeLength']();

            let infos = [];
            let info;
            for (let ii = 0; ii < channelSizeLength; ii++) {
                let dd = d['partitionSize'](ii);

                start = dd['start']();
                end = dd['end']();

                info = '[' + start.toString() + ' ,' + end.toString() + ')';
                infos.push(info);
            }

            partitionData.push(infos.join('\n'));
            partitionPos.push(positionInGroup);
        }

        return [partitionPos, partitionData];
    }

    setAttribute(layer, fused, schema) {
        // const layerType = layer.layerType();
        // const layerName = enlight.schema.LayerName[layerType];
        // const schema = this._metadata.getSchema(layerName);
        // ignore unknown layer
        const attributes = [];

        if (!schema) {
            return attributes;
        }

        const is_4bit = this.name.indexOf('4bit') > 0;
        const _layer = enlight.Node.castLayer(layer);
        const is_attr_option_required = this.getAttrOptionFlag(_layer, schema);
        if (typeof schema.bindings !== "undefined") {
            for (let i = 0 ; i < schema.bindings.length ; i++) {
                const binding = schema.bindings[i];
                const value = _layer[binding.src]();
                attributes.push(new enlight.Attribute(binding.name, value, binding.type));
            }
        }

        if (typeof schema.attributes !== "undefined") {
            for (let i = 0 ; i < schema.attributes.length ; i++) {
                const attr = schema.attributes[i];

                if (attr.name == "row_partition") {
                    const info = this.getRowPartitionInfo(_layer);
                    for (let i = 0; i < info.length; i++) {
                        attributes.push(new enlight.Attribute(`${attr.name} #${i.toString()}`, info[i], attr.type));
                    }
                }

                else if (attr.name == "och_partition") {
                    const info = this.getChannelPartitionInfo(_layer);
                    if (info) {
                        for (let i = 0; i < info[0].length; i++) {
                            attributes.push(new enlight.Attribute(`${attr.name} #${info[0][i].toString()}`, info[1][i], attr.type));
                        }
                    }
                } else {
                    // eslint-disable-next-line init-declarations
                    const value = this.packAttr(_layer, attr);
                    attributes.push(new enlight.Attribute(attr.name, value, attr.type));
                }
            }
        }

        if (typeof schema.inputs !== "undefined") {
            for (let i = 0 ; i < schema.inputs.length ; i++) {
                const input = schema.inputs[i];
                const value = _layer[input.src]();

                if (value) {
                    const args = [new enlight.Argument('', null, value, null)];
                    this.inputs.push(new enlight.Parameter(input.name, '', args));
                }
            }
        }

        if (is_attr_option_required) {
            for (let i = 0; i < schema.attributes_optional.length; i++) {
                const attr = schema.attributes_optional[i];
                const value = this.packAttr(_layer, attr);
                attributes.push(new enlight.Attribute(attr.name, value, attr.type, '', true));
            }
        }

        // eslint-disable-next-line no-negated-condition
        if (!is_4bit) {
            this._category = schema.category;
        } else {
            this._category = '4bit_layer';
        }

        return attributes;
    }

};

enlight.Attribute = class {
    constructor(name, value, type, description, visible) {
        this.name = name;
        this.value = value;
        this.visible = visible !== false;
        this.type = type;
        this.description = description || null;
    }

    get name() {
        return this.name;
    }

    get value() {
        return this.value;
    }

    get visible() {
        return this.visible;
    }
};

enlight.Parameter = class {

    constructor(name, id, args) {
        this._name = name;
        this._arguments = args;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get arguments() {
        return this._arguments;
    }

    static makeKey(layer_id, index) {
        return layer_id.toString() + "_" + index.toString();
    }
};

enlight.Argument = class {

    constructor(id, tensorInfo, initializer, stats, threshold) {
        let info = initializer ? initializer.info() : tensorInfo;

        this._id = id;
        this._type = new enlight.TensorType(info);
        this._initializer = initializer? new enlight.Tensor(info, initializer) : null;
        this._quantization = this._type.isQuantized();

        if (this._quantization) {
            this._quantization = JSON.stringify(this._type.qinfos, null, 4)
        }
        else {
            if (stats) {
                this._quantization = 'min='+stats[0].toFixed(2)+' , max='+stats[1].toFixed(2)+', mean='+stats[2].toFixed(2)+', std='+stats[3].toFixed(2);
            }

            if (threshold) {
                this._quantization += '\n\t\t\tthreshold='+threshold.toFixed(2);
            }
        }

    }

    get id() {
        return this._id;
    }

    get type() {
        return this._type;
    }

    get quantization() {
        return this._quantization;
    }

    get initializer() {
        return this._initializer;
    }
};

enlight.Tensor = class {

    constructor(tensorInfo, tensor) {
        this._name = '';
        this._type = new enlight.TensorType(tensorInfo);
        this._kind = 'ConstTensor';

        let data = null;

        if (tensor.dataType() == enlight.schema.DataType.Float32)
            data = tensor.data(new enlight.schema.FloatData);
        else if (tensor.dataType() == enlight.schema.DataType.Signed64)
            data = tensor.data(new enlight.schema.LongData);
        else if (tensor.dataType() == enlight.schema.DataType.Signed32)
            data = tensor.data(new enlight.schema.IntData);
        else if (tensor.dataType() == enlight.schema.DataType.Signed16)
            data = tensor.data(new enlight.schema.ShortData);
        else if (tensor.dataType() == enlight.schema.DataType.Signed8)
            data = tensor.data(new enlight.schema.ByteData);

        this._data = data.dataLength() > 0 ? data.dataArray() : null;
    }

    get name() {
        return this._name;
    }

    get kind() {
        return this._kind;
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state;
    }

    get value() {
        let context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        let context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        let value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        let context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;

        if (this._data == null) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        context.data = new DataView(this._data.buffer, this._data.byteOffset, this._data.byteLength);

        return context;
    }

    _decode(context, dimension) {
        let shape = context.shape;
        if (shape.length == 0) {
            shape = [ 1 ];
        }
        let size = shape[dimension];
        let results = [];
        if (dimension == shape.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (context.dataType) {
                    case 'Float16':
                        results.push(context.data.getFloat16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'Float32':
                        results.push(context.data.getFloat32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'QuantisedAsymm8':
                        results.push(context.data.getInt8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'QuantisedSymm16':
                        results.push(context.data.getInt16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'Signed8':
                        results.push(context.data.getInt8(context.index, true));
                        context.index += 1;
                        context.count++;
                        break;
                    case 'Signed16':
                        results.push(context.data.getInt16(context.index, true));
                        context.index += 2;
                        context.count++;
                        break;
                    case 'Signed32':
                        results.push(context.data.getInt32(context.index, true));
                        context.index += 4;
                        context.count++;
                        break;
                    case 'Signed64':
                        results.push('...');
                        context.index += 8;
                        context.count++;
                        break;
                    case 'Boolean':
                        results.push(context.data.getInt8(context.index));
                        context.index += 1;
                        context.count++;
                        break;
                    default:
                        break;
                }
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length == 0) {
            return results[0];
        }
        return results;
    }
};

enlight.TensorType = class {

    constructor(tensorInfo) {
        this._dataType = enlight.schema.DataTypeName[tensorInfo.dataType()] || '?';

        if (tensorInfo != null) {
            this._quantization = tensorInfo.quantizationEnabled();
        } else {
            this._quantization = false;
        }

        if (this._quantization) {
            // this.quantizationScale = tensorInfo.quantizationScale(0);
            // this.quantizationOffset = tensorInfo.quantizationOffset();
            this.qinfos = [];
            let qinfosLength = tensorInfo.quantizationScaleLength();
            if (qinfosLength> 0) {
                for (let i = 0; i < qinfosLength; i++) {
                    this.qinfos.push(tensorInfo.quantizationScale(i));
                }
            }
        }
        let dimensions = [];
        let dimensionsLength = tensorInfo.dimensionsLength();
        if (dimensionsLength > 0) {
            for (let i = 0; i < dimensionsLength; i++) {
                dimensions.push(tensorInfo.dimensions(i));
            }
        }
        this._shape = new enlight.TensorShape(dimensions);
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }

    // isQuantized() {
    //     return this._dataType.startsWith("quantised");
    // }
    isQuantized() {
        return this._quantization;
    }
};

enlight.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.map((dimension) => dimension.toString()).join(',') + ']';
    }
};

enlight.Metadata = class {
    static async open(context) {
        let data = null;
        if (!enlight.Metadata._metadata) {
            
            try {
                data = await context.request('enlight-metadata.json');
            } catch {
                // continue regardless of error
            }
            enlight.Metadata._metadata = new enlight.Metadata(data);
        }
        return new enlight.Metadata(data);
    }

    constructor(data) {
        this._map = {};
        if (data) {
            const items = JSON.parse(data);
            if (items) {
                for (const item of items) {
                    if (item.name && item.schema) {
                        this._map[item.name] = item.schema;
                    }
                }
            }
        }
    }

    getSchema(operator) {
        return this._map[operator];
    }

    getAttributeSchema(operator, name) {
        const schema = this.getSchema(operator);
        if (schema) {
            let attributeMap = schema.attributeMap;
            if (!attributeMap) {
                attributeMap = {};
                if (schema.attributes) {
                    for (const attribute of schema.attributes) {
                        attributeMap[attribute.name] = attribute;
                    }
                }
                schema.attributeMap = attributeMap;
            }
            const attributeSchema = attributeMap[name];
            if (attributeSchema) {
                return attributeSchema;
            }
        }
        return null;
    }
};

enlight.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading EnlightNN model.';
    }
};

// if (typeof module !== 'undefined' && typeof module.exports === 'object') {
//     module.exports.ModelFactory = enlight.ModelFactory;
// }
export const ModelFactory = enlight.ModelFactory;