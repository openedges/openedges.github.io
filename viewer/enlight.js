import * as enlight_pb from './enlight_pb.js';
import * as enlight_schema from './enlight-schema.js';
import * as flatbuffers from './flatbuffers-custom.js';

const enlight = {};
enlight.schema = enlight_schema.Enlight_Schema;

enlight.ModelFactory = class {

    async match(context) {
        const is_enlight = context.identifier.endsWith('enlight');

        if (is_enlight) {
            return context;
        }

        return null;
    }

    filter(context, type) {
        return true;
    }

    async open(context) {
        const file_type = (new TextDecoder()).decode(context.stream._buffer.slice(2, 18));
        if (file_type === 'enlight_protobuf') {
            return enlight_pb.EnlightV2Model.open(context);
        }
        const metadata = await enlight.Metadata.open(context);
        const container = enlight.Container.open(context, metadata);
        if (container) {
            context.type = container.type;
            context.target = container;
        }
        const target = context.target;
        await target.read();
        return new enlight.Model(metadata, target);
    }
};

enlight.Model = class {

    constructor(metadata, container) {
        this.graphs = [new enlight.Graph(metadata, container)];
        const configuration = container.configuration;
        this.name = configuration && configuration.name || "";
        this.format = container.format;
        this.metadata = [];
    }
};

enlight.Graph = class {

    constructor(metadata, container) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        const values = new Map();
        values.map = (name, type, tensor, quantization) => {
            if (name.length === 0 && tensor) {
                return new enlight.Value(name, type || null, tensor, null, quantization || null);
            }
            if (!values.has(name)) {
                values.set(name, new enlight.Value(name, type || null, tensor || null, null, quantization || null));
            } else if (tensor) {
                throw new enlight.Error(`Duplicate value '${name}'.`);
            } else if (type && !type === values.get(name).type) {
                throw new enlight.Error(`Duplicate value '${name}'.`);
            }
            return values.get(name);
        };
        // const layers = Object.entries(container.layers || {}).map(([name, value]) => {
        //     value.name = name;
        //     return value;
        // });
        const layers = container.layers;
        const inputs = new Set();
        for (const layer of layers) {
            if ("type" in layer && layer.type.name === 'Input') {
                for (let i = 0; i < layer.outputs.length; i++) {
                    const output = layer.outputs[i];
                    // const shape = Array.isArray(output.args.type.shape.dimensions) && output.args.type.shape.dimensions.length > 0 ? output.args.type.shape.dimensions : null;
                    if (!inputs.has(output)) {
                        const [output_value] = output.value;
                        const argument = new enlight.Argument('input', [values.map(output_value.name, output_value.type, null, output_value.quantization)]);
                        this.inputs.push(argument);
                        inputs.add(output);
                    }
                }
            } else if ("type" in layer && layer.type.name === 'Output') {
                for (let i = 0; i < layer.inputs.length; i++) {
                    const input = layer.inputs[i];
                    // const shape = Array.isArray(input.args.type.shape.dimensions) && input.args.type.shape.dimensions.length > 0 ? input.args.type.shape.dimensions : null;
                    const [input_value] = input.value;
                    const argument = new enlight.Argument('output', [values.map(input_value.name, input_value.type, null, input_value.quantization)]);
                    this.outputs.push(argument);
                }
            } else {
                this.nodes.push(layer);
            }
        }
    }
};

enlight.Argument = class {

    constructor(name, value, type, description, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.visible = visible !== false;
        this.description = description || null;
    }
};

enlight.Value = class {

    constructor(name, type, initializer, description, quantization) {
        if (typeof name !== 'string') {
            throw new enlight.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this.name = name;
        this.type = initializer ? initializer.type : type;
        this.initializer = initializer;
        this.description = description || null;
        this.quantization = {};
        this.quantization.type = 'lookup';
        const quantization_value = typeof quantization === 'string' ? quantization.split(', ') : [];
        this.quantization.value = quantization_value;
    }
};

enlight.Container = class {

    static open(context, metadata) {
        const identifier = context.identifier;
        const basename = identifier.split('.');
        basename.pop();
        if (identifier.toLowerCase().endsWith('.enlight')) {
            basename.pop();
            return new enlight.Container(context, 'enlight', basename.join('.'), null, metadata);
        }
        return null;
    }

    constructor(context, type, basename, configuration, metadata) {
        this.type = type;
        this.context = context;
        this.basename = basename;
        this.configuration = configuration;
        this.metadata = metadata;
        this.inputs = [];
        this.outputs = [];
        this.chain = [];
        this.layers = [];
    }

    makeKey(layer_id, index) {
        return `${layer_id.toString()}_${index.toString()}`;
    }

    async read() {
        this.format = 'enlight';
        const stream = this.context.stream;
        const buffer = stream.read(stream.length);
        const byteBuffer = new flatbuffers.ByteBuffer(buffer);
        enlight.schema = enlight_schema.Enlight_Schema;
        const model = enlight_schema.Network.prototype.getRootAsNetwork(byteBuffer);

        this.params = {};
        let paramIdx = 0;
        try {
            for (let j = 0; j < model.layersLength(); j++) {
                const base = model.layers(j).base();
                for (let i = 0 ; i < base.outputSlotsLength() ; i++) {
                    const slot = base.outputSlots(i);
                    const key = this.makeKey(base.index(), i);
                    const name = paramIdx.toString();
                    let stats = null;
                    let threshold = null;
                    if (slot.statisticsEnabled()) {
                        stats = [slot.min(), slot.max(), slot.mean(), slot.std()];
                    }
                    if (slot.thresholdEnabled()) {
                        threshold = slot.threshold();
                    }
                    const args = this.getParameter(name, slot.tensorInfo(), stats, threshold);
                    this.params[key] = { name, args };
                    paramIdx++;
                }
            }
        } catch (error) {
            let message = error && error.message ? error.message : error.toString();
            message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
            console.log(`${message} in '${this.basename}'.`);
            throw new enlight.Error(`${message} in '${this.basename}'.`);
        }
        delete this.context;
        delete this.basename;

        for (let i = 0; i < model.layersLength(); i++) {
            const layer = model.layers(i);
            const node = this.parseNode(layer, this.params, false);
            this.layers.push(node);
        }
    }

    getParameter(id, tensorInfo, stats, threshold) {
        const param = {};
        const info = tensorInfo;

        param.id = id;
        param.type = new enlight.TensorType(info);
        // param.initializer = initializer ? new enlight.Tensor(info, initializer) : null;
        param.quantization = param.type.isQuantized();

        if (param.quantization) {
            param.quantization = JSON.stringify(param.type.qinfos, null, 4);
        } else {
            if (stats) {
                param.quantization = `min=${stats[0].toFixed(2)}, max=${stats[1].toFixed(2)}, mean=${stats[2].toFixed(2)}, std=${stats[3].toFixed(2)}`;
            }

            if (threshold) {
                param.quantization += `\n\t\t\tthreshold=${threshold.toFixed(2)}`;
            }
        }
        return param;
    }

    parseNode(layer, params, fused) {
        const node = {};
        const op_type = enlight.schema.LayerName[layer.layerType()].replace(/Layer$/, '');
        node.outputs = [];
        node.inputs = [];
        node.chain = [];
        node.attributes = [];

        if (!fused) {
            const layer_base = layer.base();
            node.name = layer_base.layerName();
            node.group_idc = layer.groupIdc();
            node.position_in_group = layer.positionInGroup();

            for (let i = 0; i < layer_base.inputSlotsLength(); i++) {
                const srcConnection = layer_base.inputSlots(i).connection();
                const srcLayerIdx = srcConnection.sourceLayerIndex();
                const srcOutputIdx = srcConnection.outputSlotIndex();

                // node.inputs.push(this.params[this.makeKey(srcLayerIdx, srcOutputIdx)])
                const param = this.params[this.makeKey(srcLayerIdx, srcOutputIdx)].args;
                node.inputs.push(new enlight.Argument('X', [new enlight.Value(this.makeKey(srcLayerIdx, srcOutputIdx), param.type, null, null, param.quantization)]));
            }

            for (let j = 0; j < layer_base.outputSlotsLength(); j++) {
                // node.outputs.push(this.params[this.makeKey(layer_base.index(), j)]);
                const layer_key = this.makeKey(layer_base.index(), j);
                const param = this.params[layer_key].args;
                node.outputs.push(new enlight.Argument('Y', [new enlight.Value(layer_key, param.type, null, null, param.quantization)]));
            }

            for (let j = 0; j < layer.fusedLayersLength(); j++) {
                const fusedLayer = layer.fusedLayers(j);
                node.chain.push(this.parseNode(fusedLayer, params, true));
            }
        }

        this.setAttributes(node, layer, op_type);
        if (node.name && node.name.indexOf('4bit') > 0) {
            node.type.category = 'quantization';
        }

        return node;
    }

    setAttributes(node, layer, op_type) {
        const layerType = layer.layerType();
        const layerName = enlight.schema.LayerName[layerType];
        const schema = this.metadata.getSchema(layerName);
        node.type = {};
        node.type.name = op_type;
        node.type.category = '';
        if (!schema) {
            return;
        }

        node.type.category = schema.category;
        const castedlayer = this.castLayer(layer);
        const is_attr_option_required = this.getAttrOptionFlag(castedlayer, schema);

        if (typeof schema.bindings !== "undefined") {
            for (let i = 0 ; i < schema.bindings.length ; i++) {
                const binding = schema.bindings[i];
                const value = castedlayer[binding.src]();
                node.attributes.push(new enlight.Argument(binding.name, value, binding.type));
            }
        }

        if (typeof schema.attributes !== "undefined") {
            for (let i = 0 ; i < schema.attributes.length ; i++) {
                const attr = schema.attributes[i];
                if (attr.name === "row_partition") {
                    const info = this.getRowPartitionInfo(castedlayer);
                    for (let i = 0; i < info.length; i++) {
                        node.attributes.push(new enlight.Argument(`${attr.name} #${i.toString()}`, info[i], attr.type));
                    }
                } else if (attr.name === "och_partition") {
                    const info = this.getChannelPartitionInfo(castedlayer);
                    if (info) {
                        for (let i = 0; i < info[0].length; i++) {
                            node.attributes.push(new enlight.Argument(`${attr.name} #${info[0][i].toString()}`, info[1][i], attr.type));
                        }
                    }
                } else {
                    const value = this.packAttr(castedlayer, attr);
                    node.attributes.push(new enlight.Argument(attr.name, value, attr.type));
                }
            }
        }

        if (typeof schema.inputs !== "undefined") {
            for (let i = 0 ; i < schema.inputs.length ; i++) {
                const input = schema.inputs[i];
                const layer_info = castedlayer[input.src]();
                if (layer_info) {
                    const tensor = new enlight.Tensor(layer_info.info(), layer_info, schema.category);
                    const value = new enlight.Value(input.name, tensor.type, tensor);
                    node.inputs.push(new enlight.Argument(input.name, [value]));
                }
            }
        }

        if (is_attr_option_required) {
            for (let i = 0; i < schema.attributes_optional.length; i++) {
                const attr = schema.attributes_optional[i];
                const value = this.packAttr(castedlayer, attr);
                node.attributes.push(new enlight.Argument(attr.name, value, attr.type));
            }
        }
    }

    castLayer(layer) {
        const layerType = layer.layerType();
        for (const k of Object.keys(enlight.schema.Layer)) {
            if (layerType === enlight.schema.Layer[k]) {
                return layer.layer(new enlight.schema[k]);
            }
        }
        return null;
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
            const value = this.getAttr(descriptor, key);
            if (typeof enlight.schema[type + "Name"] != "undefined") {
                return enlight.schema[type + "Name"][value];
            } else {
                return value;
            }
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
        const descriptor = this.getDescriptor(layer);
        const partitionDescriptor = descriptor.rowPartitionData();
        if (partitionDescriptor === null) {
            return false;
        }

        const partitionData = [];
        const partitionDataLength = partitionDescriptor.rowPartitionDataLength();
        for (let i = 0; i < partitionDataLength; i++) {
            const d = partitionDescriptor.rowPartitionData(i);

            const start = d.start();
            const end = d.end();
            const info = `[${start.toString()} ,${end.toString()})`;

            partitionData.push(info);
        }

        return partitionData;
    }

    getChannelPartitionInfo(layer) {
        let descriptor = this.getDescriptor(layer);

        const partitionDescriptor = descriptor.channelPartitionData();

        if (partitionDescriptor == null) {
            return false;
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
};

enlight.Metadata = class {
    static open(host) {
        if (enlight.Metadata._metadata) {
            return Promise.resolve(enlight.Metadata._metadata);
        }
        return host.request('./enlight-metadata.json').then((data) => {
            enlight.Metadata._metadata = new enlight.Metadata(data);
            return enlight.Metadata._metadata;
        }).catch(() => {
            enlight.Metadata._metadata = new enlight.Metadata(null);
            return enlight.Metadata._metadata;
        });
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

enlight.Tensor = class {

    constructor(tensorInfo, tensor) {
        this._name = '';
        this._type = new enlight.TensorType(tensorInfo);
        let data = null;

        if (tensor.dataType() === enlight.schema.DataType.Float32) {
            data = tensor.data(new enlight.schema.FloatData);
        } else if (tensor.dataType() === enlight.schema.DataType.Signed64) {
            data = tensor.data(new enlight.schema.LongData);
        } else if (tensor.dataType() === enlight.schema.DataType.Signed32) {
            data = tensor.data(new enlight.schema.IntData);
        } else if (tensor.dataType() === enlight.schema.DataType.Signed16) {
            data = tensor.data(new enlight.schema.ShortData);
        } else if (tensor.dataType() === enlight.schema.DataType.Signed8) {
            data = tensor.data(new enlight.schema.ByteData);
        }
        this._data = data.dataLength() > 0 ? data.dataArray() : null;
        this._encoding = '|';
    }

    get name() {
        return this._name;
    }

    get category() {
        return this._category;
    }

    get type() {
        return this._type;
    }

    get state() {
        return this._context().state;
    }

    get values() {
        const context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        // return this._decode(context, 0);
        return this._decode_array(context);
    }

    get encoding() {
        return this._encoding;
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
        context.enlightDataType = this._type.enlightDataType;
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
        } else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        if (context.shape.length === 0) {
            return results[0];
        }
        return results;
    }

    _decode_array(context) {
        const shape = context.shape;
        let size = 1;
        for (let i = 0; i < shape.length; i++) {
            size *= shape[i];
        }
        const results = [];
        for (let i = 0; i < size; i++) {
            if (context.count > context.limit) {
                results.push('...');
                return results;
            }
            switch (context.enlightDataType) {
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
        return results;
    }
};

enlight.TensorType = class {

    constructor(tensorInfo) {
        this._layout = null;
        this._denotation = null;
        this._enlightDataType = enlight.schema.DataTypeName[tensorInfo.dataType()] || '?';

        if (tensorInfo != null) {
            this.quantization = tensorInfo.quantizationEnabled();
        } else {
            this.quantization = false;
        }
            
        if (this.quantization) {
            // this.quantizationScale = tensorInfo.quantizationScale(0);
            // this.quantizationOffset = tensorInfo.quantizationOffset();
            this.qinfos = [];
            const qinfosLength = tensorInfo.quantizationScaleLength();
            if (qinfosLength > 0) {
                for (let i = 0; i < qinfosLength; i++) {
                    this.qinfos.push(tensorInfo.quantizationScale(i));
                }
            }
        }
        const dimensions = [];
        const dimensionsLength = tensorInfo.dimensionsLength();
        if (dimensionsLength > 0) {
            for (let i = 0; i < dimensionsLength; i++) {
                dimensions.push(tensorInfo.dimensions(i));
            }
        }
        this._shape = new enlight.TensorShape(dimensions);
    }

    get enlightDataType() {
        return this._enlightDataType;
    }

    get dataType() {
        return this._enlightDataType.toLowerCase();
    }

    get shape() {
        return this._shape;
    }

    get denotation() {
        return this._denotation;
    }

    get layout() {
        return this._layout;
    }

    toString() {
        return this._enlightDataType + this._shape.toString();
    }

    // isQuantized() {
    //     return this._dataType.startsWith("quantised");
    // }
    isQuantized() {
        return this.quantization;
    }
};

enlight.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        if (!this.dimensions || this.dimensions.length === 0) {
            return '';
        }
        return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
    }
};

enlight.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Enlight model.';
    }
};

export const ModelFactory = enlight.ModelFactory;

