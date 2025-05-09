import * as protobuf from './protobuf.js';

const EnlightProto = {};
const EnlightV2 = {};
EnlightV2.ModelFactory = class {
    async match(context) {
        const is_enlight = context.identifier.endsWith('enlight2');
        if (is_enlight) {
            context.type = 'enlight_v2';
            context.target = EnlightProto;
            return context;
        }
        return null;
    }

    filter(context, type) {
        return true;
    }

    async open(context) {
        const model_bin = new Uint8Array(context.stream._buffer);
        const protoTest = protobuf.BinaryReader.open(model_bin);
        const header = EnlightProto.HeaderProto.decode(protoTest);
        const container =  EnlightProto.NetworkProto.decode(protoTest);
        return new EnlightV2.Model('', container);
    }
};

EnlightV2.Model = class {
    constructor(metadata, container) {
        this.graphs = [new EnlightV2.Graph(metadata, container)];
        const configuration = container.configuration;
        this.name = configuration && configuration.name || "";
        this.format = 'enlightV2';
        this.producer = 'Openedges Technology';
        this.metadata = [];
    }

    static open(context) {
        const model_bin = new Uint8Array(context.stream._buffer);
        const protoTest = protobuf.BinaryReader.open(model_bin);
        const header = EnlightProto.HeaderProto.decode(protoTest);
        const container =  EnlightProto.NetworkProto.decode(protoTest);
        return new EnlightV2.Model('', container);
    }
};

EnlightV2.Graph = class {

    constructor(metadata, container) {
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];

        for (const idx of container.inputs) {
            const tensor_info = container.tensors[idx];
            const value = new EnlightV2.Value(tensor_info, container);
            const argument = new EnlightV2.Argument('input', [value]);
            this.inputs.push(argument);
        }
        for (const idx of container.outputs) {
            const tensor_info = container.tensors[idx];
            const value = new EnlightV2.Value(tensor_info, container);
            const argument = new EnlightV2.Argument('output', [value]);
            this.outputs.push(argument);
        }
        for (const layer of container.layers) {
            this.nodes.push(new EnlightV2.Node(layer, container));
        }
    }
};

EnlightV2.Node = class {
    constructor(layer, container) {
        this.name = layer.idx;
        this.type = {};
        this.type.name = EnlightProto.LayerType.getType(layer.type);
        this.type.category = EnlightProto.LayerType.getCategory(layer.type);
        this.outputs = [];
        this.inputs = [];
        this.chain = [];
        this.attributes = [];

        switch (layer.type) {
            case EnlightProto.LayerType.LAYER_CONV:
            case EnlightProto.LayerType.LAYER_DWCONV:{
                layer.srcs.forEach((src, index) => {
                    let tensor_name = '';
                    if (index === 0) {
                        tensor_name = 'input';
                    } else if (index === 1) {
                        tensor_name = 'weight';
                    } else {
                        tensor_name = 'bias';
                    }
                    const tensor = container.tensors[src];
                    const value = new EnlightV2.Value(tensor, container);
                    const argument = new EnlightV2.Argument(tensor_name, [value]);
                    this.inputs.push(argument);
                });
                break;
            }
            case EnlightProto.LayerType.LAYER_REDUCEMAX:
            case EnlightProto.LayerType.LAYER_REDUCEMIN:
            case EnlightProto.LayerType.LAYER_REDUCEMEAN:
            case EnlightProto.LayerType.LAYER_REDUCESUM: {
                layer.srcs.forEach((src, index) => {
                    let tensor_name = '';
                    if (index === 0) {
                        tensor_name = 'input';
                    } else {
                        tensor_name = 'axes';
                    }
                    const tensor = container.tensors[src];
                    const value = new EnlightV2.Value(tensor, container);
                    const argument = new EnlightV2.Argument(tensor_name, [value]);
                    this.inputs.push(argument);
                });
                break;
            }
            default: {
                for (const src of layer.srcs) {
                    const tensor = container.tensors[src];
                    const value = new EnlightV2.Value(tensor, container);
                    const argument = new EnlightV2.Argument("", [value]);
                    this.inputs.push(argument);
                }
            }
        }

        for (const dst of layer.dsts) {
            const tensor = container.tensors[dst];
            const value = new EnlightV2.Value(tensor, container);
            const argument = new EnlightV2.Argument(dst.idx, [value]);
            this.outputs.push(argument);
        }

        for (const fused of layer.fusedLayers) {
            this.chain.push(new EnlightV2.Node(fused, container));
        }

        for (const attribute of layer.attributes) {
            this.attributes.push(new EnlightV2.Argument(attribute.name, attribute.value()));
        }
    }

};

EnlightV2.Argument = class {
    constructor(name, value, description, visible) {
        this.name = name;
        this.value = value;
        this.visible = visible !== false;
        this.description = description || null;
    }
};

EnlightV2.Value = class {
    constructor(tensor, container) {
        if (typeof name !== 'string') {
            throw new EnlightV2.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        const buffer = container.buffers[tensor.src];
        const initializer = buffer ? new EnlightV2.Initializer(tensor, buffer) : null;
        this.name = tensor.idx;
        this.type = new EnlightV2.TensorType(tensor);
        this.initializer = initializer || null;
        this.description = null;
        this.quantization = {};
        this.quantization.type = 'lookup';
        const quantization_value = typeof quantization === 'string' ? quantization.split(', ') : [];
        this.quantization.value = quantization_value;
    }
};

EnlightV2.Initializer = class {

    constructor(tensorInfo, buffer) {
        this._name = tensorInfo.idx;
        this._data = buffer.data;
        this._type = new EnlightV2.TensorType(tensorInfo);
        this._encoding = '|';
    }

    get name() {
        return this._name;
    }

    get state() {
        return this._context().state;
    }

    get type() {
        return this._type;
    }

    get values() {
        const context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        // return this._decode(context, 0);
        return this._decode(context);
    }

    get encoding() {
        return this._encoding;
    }

    _context() {
        const context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;

        if (this._data === null) {
            context.state = 'Tensor data is empty.';
            return context;
        }

        context.dataType = this._type.dataType;
        context.shape = this._type.shape.dimensions;
        context.data = this._data;
        return context;
    }

    _decode(context) {
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
            results.push(context.data[context.index]);
            context.index += 1;
            context.count++;
        }
        return results;
    }
};

EnlightV2.TensorType = class {

    constructor(tensorInfo) {
        this._layout = null;
        this._denotation = null;
        this._dataType = tensorInfo.dataType || '?';
        this.quantization = false;
        this.qinfos = [];
        this._shape = new EnlightV2.TensorShape(tensorInfo.shape);
    }

    get dataType() {
        return this._dataType.toLowerCase();
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
        return this._dataType + this._shape.toString();
    }

    isQuantized() {
        return this.quantization;
    }
};

EnlightV2.TensorShape = class {
    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length === 0) {
            return '';
        }
        return `[${this._dimensions.map((dimension) => dimension.toString()).join(',')}]`;
    }
};

EnlightV2.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading EnlightV2 model.';
    }
};

EnlightProto.HeaderProto = class {
    constructor() {
        this.file_type = '';
        this.platform = '';
        this.sdk = '';
        this.version_major = '';
        this.version_minor = '';
        this.date = '';
        this.owner = '';
    }

    static decode(reader) {
        const message = new EnlightProto.HeaderProto();
        let isRunning = true;
        while (isRunning) {
            if (reader.length <= reader.position) {
                return null;
            }

            const tag = reader.uint32();
            const field = tag >>> 3;
            switch (field) {
                case 1: {
                    message.file_type = reader.string();
                    if (message.file_type !== 'enlight_protobuf') {
                        return null;
                    }
                    break;
                }
                case 2: {
                    message.platform = reader.string();
                    break;
                }
                case 3: {
                    message.sdk = reader.string();
                    break;
                }
                case 4: {
                    message.version_major = reader.string();
                    break;
                }
                case 5: {
                    message.version_minor = reader.string();
                    break;
                }
                case 6: {
                    message.date = reader.string();
                    break;
                }
                case 7: {
                    message.owner = reader.string();
                    break;
                }
                case 32: {
                    const str = reader.string();
                    isRunning = false;
                    break;
                }
                default: {
                    reader.skipType(tag & 7);
                    break;
                }
            }
        }
        return message;
    }
};

EnlightProto.NetworkProto = class {
    constructor() {
        this.header = {};
        this.inputs = [];
        this.outputs = [];
        this.order = [];
        this.collectors = [];
        this.tensors = [];
        this.buffers = {};
        this.collectors = [];
        this.layers = [];
        this.metadata = {};
    }

    static decode(reader, length) {
        const end = length === undefined ? reader.length : reader.position + length;
        const message = new EnlightProto.NetworkProto();
        while (reader.position < end) {
            const tag = reader.uint32();
            const field = tag >>> 3;
            switch (field) {
                case 1: {
                    const read_length = reader.uint32();
                    message.header = EnlightProto.NetworkHeaderProto.decode(reader, read_length);
                    break;
                }
                case 2: {
                    const read_length = reader.uint32();
                    message.layers.push(EnlightProto.Layer.decode(reader, read_length));
                    break;
                }
                case 3: {
                    const read_length = reader.uint32();
                    const tensor = EnlightProto.Tensor.decode(reader, read_length);
                    message.tensors[tensor.idx] = tensor;
                    break;
                }
                case 4: {
                    const read_length = reader.uint32();
                    const buffer = EnlightProto.Buffer.decode(reader, read_length);
                    message.buffers[buffer.idx] = buffer;
                    break;
                }
                case 5:
                    var value = new proto.enlight.CollectorProto;
                    reader.readMessage(value, proto.enlight.CollectorProto.deserializeBinaryFromReader);
                    message.addCollectors(value);
                    break;
                case 6:
                    message.inputs.push(reader.string());
                    break;
                case 7:
                    message.outputs.push(reader.string());
                    break;
                case 8:
                    message.order.push(reader.string());
                    break;
                case 9: {
                    const read_length = reader.uint32();
                    message.metadata = EnlightProto.MetadataProto.decode(reader, read_length);
                    break;
                }
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

EnlightProto.NetworkHeaderProto = class {
    constructor() {
        this.model_name = '';
    }

    static decode(reader, length)  {
        const end = length === undefined ? reader.length : reader.position + length;
        const message = new EnlightProto.Layer();
        while (reader.position < end) {
            const tag = reader.uint32();
            const field = tag >>> 3;
            switch (field) {
                case 1:
                    message.model_name = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

EnlightProto.MetadataProto = class {
    constructor() {
        this.quantized = false;
        this.denorm_input = false;
        this.norm = [];
        this.backend = '';
    }

    static decode(reader, length)  {
        const end = length === undefined ? reader.length : reader.position + length;
        const message = new EnlightProto.MetadataProto();
        while (reader.position < end) {
            const tag = reader.uint32();
            const field = tag >>> 3;
            switch (field) {
                case 1:
                    message.quantized = reader.bool();
                    break;
                case 2:
                    message.denorm_input = reader.bool();
                    break;
                case 3:{
                    const read_length = reader.uint32();
                    message.norm.push(EnlightProto.NormInfoProto.decode(reader, read_length));
                    break;
                }
                case 4:
                    message.backend = reader.string();
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

EnlightProto.NormInfoProto = class {
    constructor() {
        this.mean = [];
        this.mean_shape = [];
        this.std = [];
        this.std_shape = [];
    }

    static decode(reader, length)  {
        const end = length === undefined ? reader.length : reader.position + length;
        const message = new EnlightProto.NormInfoProto();
        while (reader.position < end) {
            const tag = reader.uint32();
            const field = tag >>> 3;
            switch (field) {
                case 1:
                    message.mean = reader.floats(message.mean, tag);
                    break;
                case 2:
                    message.mean_shape = reader.array(message.mean_shape,
                        () => reader.int32(), tag);
                    break;
                case 3:
                    message.std = reader.floats(message.std, tag);
                    break;
                case 4:
                    message.std_shape = reader.array(message.std_shape,
                        () => reader.int32(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

EnlightProto.Layer = class {
    constructor() {
        this.idx  = '';
        this.name = '';
        this.type = '';
        this.srcs = [];
        this.dsts = [];
        this.attributes = [];
        this.fusedLayers = [];
    }

    static decode(reader, length) {
        const end = length === undefined ? reader.length : reader.position + length;
        const message = new EnlightProto.Layer();
        while (reader.position < end) {
            const tag = reader.uint32();
            const field = tag >>> 3;
            switch (field) {
                case 1:
                    message.type = reader.uint32();
                    break;
                case 2:
                    message.idx = reader.string();
                    break;
                case 3:
                    message.name = reader.string();
                    break;
                case 4: {
                    const src = reader.string();
                    if (src !== '') {
                        message.srcs.push(src);
                    }
                    break;
                }
                case 5: {
                    const dst = reader.string();
                    if (dst !== '') {
                        message.dsts.push(dst);
                    }
                    break;
                }
                case 6: {
                    const read_lenght = reader.uint32();
                    message.attributes.push(
                        EnlightProto.Attribute.decode(reader, read_lenght)
                    );
                    break;
                }
                case 7: {
                    const read_lenght = reader.uint32();
                    message.fusedLayers.push(
                        EnlightProto.FusedLayer.decode(reader, read_lenght)
                    );
                    break;
                }
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

EnlightProto.Tensor = class {
    constructor() {
        this.dataType = '';
        this.tensorType = '';
        this.idx = '';
        this.name = '';
        this.src = '';
        this.dsts = [];
        this.shape = [];
    }

    static getTensorType(value) {
        switch (value) {
            case 0:
                return 'undefined';
            case 1:
                return 'tensor';
            case 2:
                return 'initializer';
            case 3:
                return 'constant';
            default:
                return 'undefined';
        }
    }

    static decode(reader, length) {
        const end = length === undefined ? reader.length : reader.position + length;
        const message = new EnlightProto.Tensor();
        while (reader.position < end) {
            const tag = reader.uint32();
            const field = tag >>> 3;
            switch (field) {
                case 1:
                    message.dataType = EnlightProto.getDataType(reader.uint32());
                    break;
                case 2:
                    message.tensorType = EnlightProto.Tensor.getTensorType(reader.uint32());
                    break;
                case 3:
                    message.idx = reader.string();
                    break;
                case 4:
                    message.name = reader.string();
                    break;
                case 5:
                    message.src = reader.string();
                    break;
                case 6:
                    message.dsts.push(reader.string());
                    break;
                case 7:
                    message.shape = reader.array(message.shape,
                        () => reader.int32(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

EnlightProto.Buffer = class {
    constructor() {
        this.dataType = '';
        this.idx = '';
        this.name = '';
        this.dsts = [];
        this.data = [];
    }

    static decode(reader, length) {
        const message = new EnlightProto.Buffer();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            const field = tag >>> 3;
            switch (field) {
                case 1:
                    message.dataType = EnlightProto.getDataType(reader.uint32());
                    break;
                case 2:
                    message.idx = reader.string();
                    break;
                case 3:
                    message.name = reader.string();
                    break;
                case 4:
                    message.dsts.push(reader.string());
                    break;
                case 5:
                    message.data  = reader.floats(message.data, tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

EnlightProto.Attribute = class {
    constructor() {
        this.name = '';
        this.type = null;
        this.F = '';
        this.I = '';
        this.S = '';
        this.B = '';
        this.floats = [];
        this.ints = [];
        this.strings = [];
        this.bools = [];
    }

    value() {
        switch (this.type) {
            case 1:
                return this.F;
            case 2:
                return this.I;
            case 3:
                return this.S;
            case 4:
                return this.B.toString();
            case 5:
                return this.floats;
            case 6:
                return this.ints;
            case 7:
                return this.strings;
            case 8:
                return this.bools;
            default:
                return '';
        }
    }

    static decode(reader, length) {
        const message = new EnlightProto.Attribute();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            const field = tag >>> 3;
            switch (field) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = reader.uint32();
                    break;
                case 3:
                    message.F = reader.float();
                    break;
                case 4:
                    message.I = reader.int32();
                    break;
                case 5:
                    message.S = reader.string();
                    break;
                case 6:
                    message.B = reader.bool();
                    break;
                case 7:
                    message.floats = reader.floats(message.floats, tag);
                    break;
                case 8:
                    message.ints = reader.array(message.ints,
                        () => reader.int32(), tag);
                    break;
                case 9:
                    message.strings = reader.array(message.strings,
                        () => reader.string(), tag);
                    break;
                case 10:
                    message.bools = reader.array(message.bools, () => reader.bool(), tag);
                    break;
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

EnlightProto.FusedLayer = class {
    constructor() {
        this.name = '';
        this.type = '';
        this.attributes = [];
        this.srcs = [];
        this.dsts = [];
        this.fusedLayers = [];
    }

    static decode(reader, length) {
        const message = new EnlightProto.FusedLayer();
        const end = length === undefined ? reader.length : reader.position + length;
        while (reader.position < end) {
            const tag = reader.uint32();
            const field = tag >>> 3;
            switch (field) {
                case 1:
                    message.name = reader.string();
                    break;
                case 2:
                    message.type = reader.uint32();
                    break;
                case 3: {
                    const read_lenght = reader.uint32();
                    message.attributes.push(
                        EnlightProto.Attribute.decode(reader, read_lenght)
                    );
                    break;
                }
                default:
                    reader.skipType(tag & 7);
                    break;
            }
        }
        return message;
    }
};

EnlightProto.LayerType = {
    LAYER_UNDEFINED: 0,
    LAYER_RELU: 1,
    LAYER_SIGMOID: 2,
    LAYER_LEAKYRELU: 3,
    LAYER_TANH: 4,
    LAYER_RELU6: 5,
    LAYER_CLIP: 6,
    LAYER_HARDSIGMOID: 7,
    LAYER_HARDSWISH: 8,
    LAYER_SWISH: 9,
    LAYER_MISH: 10,
    LAYER_MISHATTENTION: 11,
    LAYER_PRELU: 12,
    LAYER_SOFTPLUS: 13,
    LAYER_ERF: 14,
    LAYER_GELU: 15,
    LAYER_LINEARAPPROX: 16,
    LAYER_LINEARAPPROXLU: 17,
    LAYER_ADD: 18,
    LAYER_SUB: 19,
    LAYER_MUL: 20,
    LAYER_DIV: 21,
    LAYER_EXP: 22,
    LAYER_LOG: 23,
    LAYER_FLOOR: 24,
    LAYER_CEIL: 25,
    LAYER_ROUND: 26,
    LAYER_POW: 27,
    LAYER_SQDIFF: 28,
    LAYER_RECIPROCAL: 29,
    LAYER_MEAN: 30,
    LAYER_SUM: 31,
    LAYER_SQRT: 32,
    LAYER_RSQRT: 33,
    LAYER_L2NORM: 34,
    LAYER_NOT: 35,
    LAYER_CUMSUM: 36,
    LAYER_SIN: 37,
    LAYER_COS: 38,
    LAYER_TILE: 39,
    LAYER_WHERE: 40,
    LAYER_EQUAL: 41,
    LAYER_MULCONST: 42,
    LAYER_CONV: 43,
    LAYER_DWCONV: 44,
    LAYER_CONVTRANSPOSE: 45,
    LAYER_CONVTRANSPOSEWITHZEROFILL: 46,
    LAYER_BATCHNORM: 47,
    LAYER_MAXPOOL: 48,
    LAYER_AVGPOOL: 49,
    LAYER_GLOBALAVGPOOL: 50,
    LAYER_MATMUL: 51,
    LAYER_GEMM: 52,
    LAYER_LINEAR: 53,
    LAYER_RESIZE: 54,
    LAYER_REDUCEMAX: 55,
    LAYER_REDUCEMEAN: 56,
    LAYER_REDUCEMIN: 57,
    LAYER_REDUCEPROD: 58,
    LAYER_REDUCESUM: 59,
    LAYER_SOFTMAX: 60,
    LAYER_LOGSOFTMAX: 61,
    LAYER_DETECTIONPOSTPROCESS: 62,
    LAYER_IDENTITY: 63,
    LAYER_LAYERNORM: 64,
    LAYER_DROPOUT: 65,
    LAYER_DEPTHTOSPACE: 66,
    LAYER_SPACETODEPTH: 67,
    LAYER_QLINEARCONV: 68,
    LAYER_QLINEARMATMUL: 69,
    LAYER_QLINEAR: 70,
    LAYER_DQLINEAR: 71,
    LAYER_BYPASS: 72,
    LAYER_CONCAT: 73,
    LAYER_FLATTEN: 74,
    LAYER_GATHER: 75,
    LAYER_GATHERELEMENT: 76,
    LAYER_GATHERND: 77,
    LAYER_PACK: 78,
    LAYER_UNPACK: 79,
    LAYER_PAD: 80,
    LAYER_MADD: 81,
    LAYER_RESHAPE: 82,
    LAYER_SHAPE: 83,
    LAYER_SLICE: 84,
    LAYER_UNSQUEEZE: 85,
    LAYER_SQUEEZE: 86,
    LAYER_TRANSPOSE: 87,
    LAYER_CAST: 88,
    LAYER_CONSTANTOFSHAPE: 89,
    LAYER_CONSTANT: 90,
    LAYER_EXPAND: 91,
    LAYER_SIZE: 92,
    LAYER_SPLIT: 93,
    LAYER_RANGE: 94,
    LAYER_QUANT: 95,
    LAYER_DEQUANT: 96,
    LAYER_REQUANT: 97,
    LAYER_RESCALE: 98,
    LAYER_DELIMITER: 99,
    LAYER_DEQUANTV2: 100,
    LAYER_REQUANTV2: 101,
    LAYER_CHALIGN: 102,
    LAYER_ROWIM2COL: 103,
    LAYER_COLIM2COL: 104,
    LAYER_DATALAYOUTCONVERSION_VEC2MM: 105,
    LAYER_DATALAYOUTCONVERSION_GBUF2HWC: 106,
    LAYER_DATALAYOUTCONVERSION_HWC2GBUF: 107,
    LAYER_DATALAYOUTCONVERSION_MM2VEC: 108,
    LAYER_ELTWSCALAR: 109,
    LAYER_CONCATALIGN: 110,
    LAYER_ROWCONCAT: 111,
    LAYER_ADDCONST: 112,
    LAYER_ROWCONCATCONST: 113,
    LAYER_SCALEDOTPRODUCTATTENTION: 114,
    LAYER_SPLITHEAD: 115,
    LAYER_MERGEHEAD: 116,
    LAYER_UNKNOWN: 117,
    LAYER_ANY: 118
};

EnlightProto.LayerType.list = Object.keys(EnlightProto.LayerType).map((type) => type.replace("LAYER_", ""));
EnlightProto.LayerType.getType = function(type) {
    return EnlightProto.LayerType.list[type] || '';
};

EnlightProto.LayerType.getCategory = function(type) {
    const activation = [
        EnlightProto.LayerType.LAYER_RELU,
        EnlightProto.LayerType.LAYER_SIGMOID,
        EnlightProto.LayerType.LAYER_LEAKYRELU,
        EnlightProto.LayerType.LAYER_TANH,
        EnlightProto.LayerType.LAYER_RELU6,
        EnlightProto.LayerType.LAYER_CLIP,
        EnlightProto.LayerType.LAYER_HARDSIGMOID,
        EnlightProto.LayerType.LAYER_HARDSWISH,
        EnlightProto.LayerType.LAYER_SWISH,
        EnlightProto.LayerType.LAYER_MISH,
        EnlightProto.LayerType.LAYER_MISHATTENTION,
        EnlightProto.LayerType.LAYER_PRELU,
        EnlightProto.LayerType.LAYER_SOFTPLUS,
        EnlightProto.LayerType.LAYER_ERF,
        EnlightProto.LayerType.LAYER_GELU,
        EnlightProto.LayerType.LAYER_LINEARAPPROX,
        EnlightProto.LayerType.LAYER_LINEARAPPROXLU,
        EnlightProto.LayerType.LAYER_SOFTMAX,
        EnlightProto.LayerType.LAYER_LOGSOFTMAX
    ];

    const normalization = [
        EnlightProto.LayerType.LAYER_L2NORM,
        EnlightProto.LayerType.LAYER_LAYERNORM,
        EnlightProto.LayerType.LAYER_BATCHNORM
    ];

    const arithmetic = [
        EnlightProto.LayerType.LAYER_ADD,
        EnlightProto.LayerType.LAYER_SUB,
        EnlightProto.LayerType.LAYER_MUL,
        EnlightProto.LayerType.LAYER_DIV,
        EnlightProto.LayerType.LAYER_EXP,
        EnlightProto.LayerType.LAYER_LOG,
        EnlightProto.LayerType.LAYER_FLOOR,
        EnlightProto.LayerType.LAYER_CEIL,
        EnlightProto.LayerType.LAYER_ROUND,
        EnlightProto.LayerType.LAYER_POW,
        EnlightProto.LayerType.LAYER_SQDIFF,
        EnlightProto.LayerType.LAYER_RECIPROCAL,
        EnlightProto.LayerType.LAYER_MEAN,
        EnlightProto.LayerType.LAYER_SUM,
        EnlightProto.LayerType.LAYER_SQRT,
        EnlightProto.LayerType.LAYER_RSQRT,
        EnlightProto.LayerType.LAYER_NOT,
        EnlightProto.LayerType.LAYER_CUMSUM,
        EnlightProto.LayerType.LAYER_SIN,
        EnlightProto.LayerType.LAYER_COS,
        EnlightProto.LayerType.LAYER_TILE,
        EnlightProto.LayerType.LAYER_EQUAL,
        EnlightProto.LayerType.LAYER_MULCONST,
        EnlightProto.LayerType.LAYER_ADDCONST
    ];

    const pool = [
        EnlightProto.LayerType.LAYER_MAXPOOL,
        EnlightProto.LayerType.LAYER_AVGPOOL,
        EnlightProto.LayerType.LAYER_GLOBALAVGPOOL,
        EnlightProto.LayerType.LAYER_REDUCEMAX,
        EnlightProto.LayerType.LAYER_REDUCEMEAN,
        EnlightProto.LayerType.LAYER_REDUCEMIN,
        EnlightProto.LayerType.LAYER_REDUCEPROD,
        EnlightProto.LayerType.LAYER_REDUCESUM
    ];

    const layer = [
        EnlightProto.LayerType.LAYER_CONV,
        EnlightProto.LayerType.LAYER_DWCONV,
        EnlightProto.LayerType.LAYER_CONVTRANSPOSE,
        EnlightProto.LayerType.LAYER_CONVTRANSPOSEWITHZEROFILL,
        EnlightProto.LayerType.LAYER_MATMUL,
        EnlightProto.LayerType.LAYER_GEMM,
        EnlightProto.LayerType.LAYER_LINEAR,
        EnlightProto.LayerType.LAYER_QLINEARCONV,
        EnlightProto.LayerType.LAYER_QLINEARMATMUL,
        EnlightProto.LayerType.LAYER_QLINEAR,
        EnlightProto.LayerType.LAYER_DQLINEAR,
        EnlightProto.LayerType.LAYER_BYPASS,
        EnlightProto.LayerType.LAYER_MADD,
        EnlightProto.LayerType.LAYER_ELTWSCALAR,
        EnlightProto.LayerType.LAYER_DEPTHTOSPACE,
        EnlightProto.LayerType.LAYER_SPACETODEPTH,
        EnlightProto.LayerType.LAYER_DELIMITER,
        EnlightProto.LayerType.LAYER_DETECTIONPOSTPROCESS,
    ];

    const dropout = [
        EnlightProto.LayerType.LAYER_DROPOUT
    ];

    const shape = [
        EnlightProto.LayerType.LAYER_RESHAPE,
        EnlightProto.LayerType.LAYER_RESIZE,
        EnlightProto.LayerType.LAYER_UNSQUEEZE,
        EnlightProto.LayerType.LAYER_SQUEEZE,
        EnlightProto.LayerType.LAYER_TRANSPOSE,
        EnlightProto.LayerType.LAYER_EXPAND
    ];

    const tensor = [
        EnlightProto.LayerType.LAYER_PACK,
        EnlightProto.LayerType.LAYER_UNPACK,
        EnlightProto.LayerType.LAYER_PAD,
        EnlightProto.LayerType.LAYER_CONCAT,
        EnlightProto.LayerType.LAYER_SLICE,
        EnlightProto.LayerType.LAYER_FLATTEN,
        EnlightProto.LayerType.LAYER_GATHER,
        EnlightProto.LayerType.LAYER_GATHERELEMENT,
        EnlightProto.LayerType.LAYER_GATHERND,
        EnlightProto.LayerType.LAYER_SPLIT,
        EnlightProto.LayerType.LAYER_RANGE,
        EnlightProto.LayerType.LAYER_SIZE,
        EnlightProto.LayerType.LAYER_SHAPE,
        EnlightProto.LayerType.LAYER_ROWCONCAT,
        EnlightProto.LayerType.LAYER_ROWCONCATCONST,
        EnlightProto.LayerType.LAYER_CONCATALIGN,
        EnlightProto.LayerType.LAYER_CHALIGN,
        EnlightProto.LayerType.LAYER_ROWIM2COL,
        EnlightProto.LayerType.LAYER_COLIM2COL,
        EnlightProto.LayerType.LAYER_DATALAYOUTCONVERSION_VEC2MM,
        EnlightProto.LayerType.LAYER_DATALAYOUTCONVERSION_GBUF2HWC,
        EnlightProto.LayerType.LAYER_DATALAYOUTCONVERSION_HWC2GBUF,
        EnlightProto.LayerType.LAYER_DATALAYOUTCONVERSION_MM2VEC,
        EnlightProto.LayerType.LAYER_IDENTITY,
        EnlightProto.LayerType.LAYER_CAST
    ];

    const quantization = [
        EnlightProto.LayerType.LAYER_QUANT,
        EnlightProto.LayerType.LAYER_DEQUANT,
        EnlightProto.LayerType.LAYER_REQUANT,
        EnlightProto.LayerType.LAYER_RESCALE,
        EnlightProto.LayerType.LAYER_DEQUANTV2,
        EnlightProto.LayerType.LAYER_REQUANTV2,
    ];

    const attention = [
        EnlightProto.LayerType.LAYER_SPLITHEAD,
        EnlightProto.LayerType.LAYER_MERGEHEAD,
        EnlightProto.LayerType.LAYER_SCALEDOTPRODUCTATTENTION
    ];

    const constant = [
        EnlightProto.LayerType.LAYER_CONSTANT,
        EnlightProto.LayerType.LAYER_CONSTANTOFSHAPE
    ];

    if (activation.includes(type)) {
        return 'activation';
    } else if (normalization.includes(type)) {
        return 'normalization';
    } else if (layer.includes(type) || arithmetic.includes(type)) {
        return 'layer';
    } else if (pool.includes(type)) {
        return 'pool';
    } else if (shape.includes(type)) {
        return 'shape';
    } else if (dropout.includes(type)) {
        return 'dropout';
    } else if (quantization.includes(type)) {
        return 'quantization';
    } else if (constant.includes(type)) {
        return 'constant';
    } else if (attention.includes(type)) {
        return 'attention';
    } else if (tensor.includes(type)) {
        return 'tensor';
    }
    return '';
};

EnlightProto.AttributeType = {
    ATTR_UNDEFINED: 0,
    ATTR_F: 1,
    ATTR_I: 2,
    ATTR_S: 3,
    ATTR_B: 4,
    ATTR_FLOATS: 5,
    ATTR_INTS: 6,
    ATTR_STRINGS: 7,
    ATTR_BOOLS: 8,
    ATTR_FUSED_ANY: 9,
    ATTR_FUSED_ACT: 10
};

EnlightProto.getDataType = function(value) {
    switch (value) {
        case 0:
            return 'none';
        case 1:
            return 'uint8';
        case 2:
            return 'uint16';
        case 3:
            return 'uint32';
        case 4:
            return 'uint64';
        case 5:
            return 'int8';
        case 6:
            return 'int16';
        case 7:
            return 'int32';
        case 8:
            return 'int64';
        case 9:
            return 'float8';
        case 10:
            return 'float16';
        case 11:
            return 'float32';
        case 12:
            return 'double';
        case 13:
            return 'bool';
        default:
            return 'undefined';
    }
};

EnlightProto.DataType = {
    DATA_UNDEFINED: 0,
    DATA_NONE: 0,
    DATA_UINT8: 1,
    DATA_UINT16: 2,
    DATA_UINT32: 3,
    DATA_UINT64: 4,
    DATA_INT8: 5,
    DATA_INT16: 6,
    DATA_INT32: 7,
    DATA_INT64: 8,
    DATA_FLOAT8: 9,
    DATA_FLOAT16: 10,
    DATA_FLOAT32: 11,
    DATA_FLOAT: 11,
    DATA_FLOAT64: 12,
    DATA_DOUBLE: 12,
    DATA_BOOL: 13
};

export const ModelFactory = EnlightV2.ModelFactory;
export const EnlightV2Model = EnlightV2.Model;