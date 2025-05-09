import * as enlight_schema from './enlight-schema.js';
import * as flatbuffers from './flatbuffers-custom.js';

const enlight = {};
enlight.schema = enlight_schema.Enlight_Schema;
enlight.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extensions = ['enlight',];
        if (extensions.some((extension) => identifier.endsWith(extension))) {
            const entries = [
                enlight.FlatReader
            ];
            for (const entry of entries) {
                const reader = entry.open(context);
                if (reader) {
                    context.type = reader.name;
                    context.target = reader;
                    break;
                }
            }
        }
    }

    async open(context) {
        const target = context.target;
        await target.read();
        const metadata = await enlight.Metadata.open(context);
        return new enlight.Model(target, metadata);
    }

    filter(context, type) {
        return context.type !== 'onnx.proto' || (type !== 'onnx.data' && type !== 'dot');
    }
};

enlight.Model = class {

    constructor(target, metadata) {
        const model = target.model;
        this._graphs = [];
        this._format = target.format;
        this._producer = model.producer_name && model.producer_name.length > 0 ? model.producer_name + (model.producer_version && model.producer_version.length > 0 ? ` ${model.producer_version}` : '') : null;
        this._domain = model.domain;
        this._version = typeof model.model_version === 'number' || typeof model.model_version === 'bigint' ? model.model_version.toString() : '';
        this._description = model.doc_string;
        this._metadata = [];
        this._imports = null;
        const imports = new Map();
        if (model.opset_import && model.opset_import.length > 0) {
            for (const opset_import of model.opset_import) {
                const domain = opset_import.domain || 'ai.onnx';
                const version = typeof opset_import.version === 'bigint' ? opset_import.version.toNumber() : opset_import.version;
                if (!imports.has(domain) || imports.get(domain) > version) {
                    imports.set(domain, version);
                }
            }
            this._imports = Array.from(imports).map(([name, version]) => `${name} v${version}`);
        }
        if (imports.size === 0) {
            imports.set('ai.onnx', 1);
            imports.set('ai.onnx.ml', 1);
        }
        let imageFormat = '';
        const metadata_props = model.metadata_props;
        if (metadata_props) {
            const metadata = new Map(metadata_props.map((entry) => [entry.key, entry.value]));
            const converted_from = metadata.get('converted_from');
            if (converted_from) {
                this.source = converted_from;
            }
            const author = metadata.get('author');
            if (author) {
                this._metadata.push(new enlight.Argument('author', author));
            }
            const company = metadata.get('company');
            if (company) {
                this._metadata.push(new enlight.Argument('company', company));
            }
            let license = metadata.get('license');
            const license_url = metadata.get('license_url');
            if (license_url) {
                license = `<a href='${license_url}'>${license ? license : license_url}</a>`;
            }
            if (license) {
                this._metadata.push(new enlight.Argument('license', license));
            }
            metadata.delete('author');
            metadata.delete('company');
            metadata.delete('converted_from');
            metadata.delete('license');
            metadata.delete('license_url');
            const imageMetadata = {};
            for (const [name, value] of metadata) {
                switch (name) {
                    case 'Image.BitmapPixelFormat':
                    case 'Image.ColorSpaceGamma':
                    case 'Image.NominalPixelRange':
                        imageMetadata[name] = value;
                        break;
                    default:
                        this._metadata.push(new enlight.Argument(name, value));
                        break;
                }
            }
            imageFormat = [imageMetadata['Image.BitmapPixelFormat'], imageMetadata['Image.ColorSpaceGamma'], imageMetadata['Image.NominalPixelRange']].filter((item) => item);
        }
        const context = new enlight.Context.Model(metadata, target.locations, imageFormat, imports, model, model.functions);
        if (context.graph) {
            this._graphs.push(context.graph);
        }
    }

    get format() {
        return this._format;
    }

    get version() {
        return this._version;
    }

    get imports() {
        return this._imports;
    }

    get producer() {
        return this._producer;
    }

    get source() {
        return this._source;
    }

    get domain() {
        return this._domain || null;
    }

    get description() {
        return this._description || null;
    }

    get metadata() {
        return this._metadata;
    }

    get graphs() {
        return this._graphs;
    }
};

enlight.Graph = class {

    constructor(context, graph) {
        this._description = '';
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];
        this._name = graph && graph.name || null;
        this._description = graph.doc_string || '';
        // context = new enlight.Context.Graph(context, graph);
        // if (Array.isArray(graph.quantization_annotation)) {
        //     for (const tensor_annotation of graph.quantization_annotation) {
        //         const tensor = context.tensor(tensor_annotation.tensor_name);
        //         tensor.annotation = new Map();
        //         for (const entry of tensor_annotation.quant_parameter_tensor_names) {
        //             tensor.annotation.set(entry.key, entry.value);
        //         }
        //     }
        // }
        // if (Array.isArray(graph.value_info)) {
        //     for (const value of graph.value_info) {
        //         const tensor = context.tensor(value.name);
        //         tensor.type = context.createType(value.type);
        //         tensor.description = value.doc_string;
        //     }
        // }

        this._name = '';
        this._nodes = [];
        this._inputs = [];
        this._outputs = [];

        let params = {};

        // generate parameters
        let paramIdx = 0;
        for (let j = 0; j < graph.layersLength(); j++) {
            let base = enlight.Context.Node.getBase(graph.layers(j));
            for (let i = 0 ; i < base.outputSlotsLength() ; i++) {
                let slot = base.outputSlots(i);
                let key = enlight.Context.Parameter.makeKey(base.index(), i);
                let name = paramIdx.toString();

                let stats = null;
                let threshold = null;

                if (slot.statisticsEnabled()) {
                    stats = [slot.min(), slot.max(), slot.mean(), slot.std()];
                }

                if (slot.thresholdEnabled()) {
                    threshold = slot.threshold();
                }

                let args = [new enlight.Context.Argument(name, slot.tensorInfo(), null, stats, threshold)];
                params[key] = new enlight.Context.Parameter(name, name, args);
                paramIdx++;
            }
        }

        // generate nodes
        for (let j = 0; j < graph.layersLength(); j++) {
            const node = new enlight.Context.Node(graph.layers(j), params, false);
            console.log(node);
            // this._nodes.push(new enlight.Context.Node(graph.layers(j), params, false));
        }

        graph.input = graph.input.map((value) => {
            const tensor = context.tensor(value.name);
            tensor.type = context.createType(value.type);
            tensor.description = value.doc_string;
            return tensor;
        });
        graph.output = graph.output.map((value) => {
            const tensor = context.tensor(value.name);
            tensor.type = context.createType(value.type);
            tensor.description = value.doc_string;
            return tensor;
        });
        const inference = new enlight.Inference(graph.node);
        for (const output of graph.output) {
            inference.infer(output.name);
        }
        context.push(graph.node, graph.input, graph.output);
        this._nodes = context.pop();
        for (const input of graph.input) {
            const value = context.value(input.name);
            if (!value.initializer) {
                this._inputs.push(new enlight.Argument(input.name, [value]));
            }
        }
        for (const output of graph.output) {
            const value = context.value(output.name);
            if (!value.initializer) {
                this._outputs.push(new enlight.Argument(output.name, [value]));
            }
        }
        const metadata_props = graph.metadata_props || [];
        this.metadata = metadata_props.map((metadata) => {
            return new enlight.Argument(metadata.key, metadata.value);
        });
    }

    get name() {
        return this._name;
    }

    get description() {
        return this._description;
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

    toString() {
        return `graph(${this.name})`;
    }
};

enlight.Argument = class {

    constructor(name, value, type, description, visible) {
        this.name = name;
        this.value = value;
        this.type = type || null;
        this.description = description || null;
        this.visible = visible !== false;
    }
};

enlight.Value = class {

    constructor(name, type, initializer, annotation, description) {
        if (typeof name !== 'string') {
            throw new enlight.Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
        }
        this._name = name;
        this._type = type || null;
        this._initializer = initializer || null;
        this._description = description || '';
        this._quantization = annotation ? { type: 'annotation', value: annotation } : null;
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get description() {
        return this._description;
    }

    get quantization() {
        return this._quantization;
    }

    get initializer() {
        return this._initializer;
    }
};

enlight.Node = class {

    constructor(context, node) {
        const attributes = node.attribute || [];
        const metadata_props = node.metadata_props || [];
        const domain = node.domain || 'ai.onnx';
        let op_type = node.op_type;
        let overload = node.overload || '';
        if (domain === 'pkg.torch.ops') {
            const path = op_type.split('.');
            overload = path.pop();
            op_type = path.join('.');
        }
        this.type = context.type(domain, op_type, overload);
        if (!this.type || (this.type.module !== domain && !(this.type instanceof enlight.Function))) {
            this.type = { ...this.type };
            this.type.name = op_type;
            this.type.module = domain;
            this.type.overload = overload;
            this.type.identifier = overload ? `${op_type}.${overload}` : `${op_type}`;
        }
        this.metadata = [];
        for (const metadata of metadata_props) {
            const key = metadata.key;
            const value = metadata.value;
            if (key === 'input_names' && value.startsWith('[') && value.endsWith(']') && !Array.isArray(this.type.inputs)) {
                const input_names = value.slice(1, -1).split(', ');
                if (input_names.every((item) => /^'.*'$/.test(item))) {
                    this.type.inputs = input_names.map((item) => ({ name: item.slice(1, -1) }));
                    continue;
                }
            }
            const argument = new onnx.Argument(metadata.key, metadata.value);
            this.metadata.push(argument);
        }
        const inputs = [];
        node.input = node.input || [];
        for (let i = 0; i < node.input.length;) {
            const input = this.type && Array.isArray(this.type.inputs) && i < this.type.inputs.length ? this.type.inputs[i] : { name: i.toString() };
            const count = input.list ? node.input.length - i : 1;
            const list = node.input.slice(i, i + count).filter((value) => value.name !== '' || value.initializer);
            const values = list.map((input) => context.value(input.name));
            const argument = new onnx.Argument(input.name, values);
            inputs.push(argument);
            i += count;
        }
        const outputs = [];
        node.output = node.output || [];
        for (let i = 0; i < node.output.length;) {
            const output = this.type && Array.isArray(this.type.outputs) && i < this.type.outputs.length ? this.type.outputs[i] : { name: i.toString() };
            const count = output.list ? node.output.length - i : 1;
            const list = node.output.slice(i, i + count).filter((value) => value.name !== '' || value.initializer);
            const values = list.map((output) => context.value(output.name));
            const argument = new onnx.Argument(output.name, values);
            outputs.push(argument);
            i += count;
        }
        this.name = node.name || '';
        this.description = node.doc_string || '';
        this.inputs = inputs || [];
        this.outputs = outputs || [];
        this.attributes = attributes.map((attribute) => {
            const name = attribute.name;
            let type = null;
            let value = null;
            let visible = true;
            if (attribute.ref_attr_name) {
                value = attribute.ref_attr_name;
                type = 'reference';
            } else {
                switch (attribute.type) {
                    case onnx.AttributeType.UNDEFINED:
                        break;
                    case onnx.AttributeType.FLOAT:
                        value = attribute.f;
                        type = 'float32';
                        break;
                    case onnx.AttributeType.INT:
                        value = BigInt(attribute.i);
                        type = 'int64';
                        break;
                    case onnx.AttributeType.STRING:
                        value = op_type === 'Int8GivenTensorFill' ? Array.from(attribute.s) : context.decodeText(attribute.s);
                        type = 'string';
                        break;
                    case onnx.AttributeType.TENSOR:
                        value = new onnx.Tensor(context, attribute.t);
                        type = 'tensor';
                        break;
                    case onnx.AttributeType.GRAPH:
                        value = context.graph(attribute.g);
                        type = 'graph';
                        break;
                    case onnx.AttributeType.FLOATS:
                        value = ArrayBuffer.isView(attribute.floats) ? Array.from(attribute.floats) : attribute.floats;
                        type = 'float32[]';
                        break;
                    case onnx.AttributeType.INTS:
                        value = ArrayBuffer.isView(attribute.ints) ? Array.from(attribute.ints) : attribute.ints.map((value) => BigInt(value));
                        type = 'int64[]';
                        break;
                    case onnx.AttributeType.STRINGS:
                        value = attribute.strings.map((s) => context.decodeText(s));
                        type = 'string[]';
                        break;
                    case onnx.AttributeType.TENSORS:
                        value = attribute.tensors.map((tensor) => new onnx.Tensor(context, tensor));
                        type = 'tensor[]';
                        break;
                    case onnx.AttributeType.GRAPHS:
                        value = attribute.graphs.map((graph) => context.graph(graph));
                        type = 'graph[]';
                        break;
                    case onnx.AttributeType.SPARSE_TENSOR:
                        value = new onnx.Tensor(context, attribute.sparse_tensor);
                        type = 'tensor';
                        break;
                    case onnx.AttributeType.SPARSE_TENSORS:
                        value = attribute.sparse_tensors.map((tensor) => new onnx.Tensor(context, tensor));
                        type = 'tensor[]';
                        break;
                    case onnx.AttributeType.TYPE_PROTO:
                        value = context.createType(attribute.tp);
                        type = 'type';
                        break;
                    case onnx.AttributeType.TYPE_PROTOS:
                        value = attribute.type_protos.map((type) => context.createType(type));
                        type = 'type[]';
                        break;
                    default:
                        throw new onnx.Error(`Unsupported attribute type '${attribute.type}'.`);
                }
                const metadata = context.attribute(domain, op_type, overload, attribute.name);
                if (metadata) {
                    if (metadata.default !== undefined) {
                        const defaultValue = type === 'int64' ? BigInt(metadata.default) : metadata.default;
                        if (value === defaultValue) {
                            visible = false;
                        }
                    }
                    if (metadata.type === 'DataType') {
                        type = metadata.type;
                        value = context.createDataType(value);
                    }
                }
            }
            return new onnx.Argument(name, value, type, attribute.doc_string, visible);
        });
        this.chain = [];
        const identifier = domain ? `${domain}.${op_type}` : op_type;
        if (identifier === 'com.microsoft.FusedConv') {
            const activation = attributes.find((attribute) => attribute.name === 'activation');
            if (activation) {
                const type = context.decodeText(activation.s);
                const node = new onnx.Node(context, { op_type: type });
                this.chain.push(node);
            }
        }
    }
};

enlight.Group = class {

    constructor(name, groups) {
        this._type = { name: 'Scope' };
        this._name = name;
        this._nodes = [];
        for (const [key, value] of groups) {
            if (key === '') {
                for (const node of value) {
                    this._nodes.push(node);
                }
            } else {
                this._nodes.push(new onnx.Group(name === '' ? key : `${name}/${key}`, value));
            }
        }
        const set = new Set();
        const inputs = [];
        const outputs = [];
        for (const node of this._nodes) {
            if (node instanceof onnx.Group) {
                node.freeze();
            }
            for (const parameter of node.outputs) {
                for (const value of parameter.value) {
                    if (!value.initializer) {
                        outputs.push(value);
                        set.add(value.name);
                    }
                }
            }
        }
        for (const node of this._nodes) {
            for (const parameter of node.inputs) {
                for (const value of parameter.value) {
                    if (!set.has(value.name) && !value.initializer) {
                        inputs.push(value);
                    }
                }
            }
        }
        this._inputs = [new onnx.Argument('inputs', inputs)];
        this._outputs = [new onnx.Argument('outputs', outputs)];
        this._attributes = [];
    }

    get name() {
        return this._name;
    }

    get type() {
        return this._type;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get attributes() {
        return this._attributes;
    }

    get nodes() {
        return this._nodes;
    }
};

enlight.Tensor = class {

    constructor(context, tensor, category) {
        this._category = category || null;
        if (tensor.indices && tensor.values) {
            this._name = tensor.values.name || '';
            this._type = context.createTensorType(tensor.values.data_type, tensor.dims, 'sparse');
            this._location = context.createLocation(tensor.values.data_location);
            this._values = new onnx.Tensor(context, tensor.values);
            this._indices = new onnx.Tensor(context, tensor.indices);
        } else {
            this._name = tensor.name || '';
            this._type = context.createTensorType(tensor.data_type, tensor.dims);
            this._location = context.createLocation(tensor.data_location);
            switch (tensor.data_location) {
                case onnx.DataLocation.DEFAULT: {
                    switch (tensor.data_type) {
                        case onnx.DataType.UNDEFINED: {
                            break;
                        }
                        case onnx.DataType.FLOAT:
                            this._data = new Float32Array(tensor.float_data);
                            this._encoding = '|';
                            break;
                        case onnx.DataType.DOUBLE:
                            this._data = new Float64Array(tensor.double_data);
                            this._encoding = '|';
                            break;
                        case onnx.DataType.BOOL:
                            if (tensor.int32_data && tensor.int32_data.length > 0) {
                                const array = tensor.int32_data;
                                this._data = new Array(array.length);
                                for (let i = 0; i < this._data.length; i++) {
                                    this._data[i] = array[i] === 0 ? false : true;
                                }
                                this._encoding = '|';
                            }
                            break;
                        case onnx.DataType.INT8:
                            this._data = new Int8Array(tensor.int32_data);
                            this._encoding = '|';
                            break;
                        case onnx.DataType.UINT8:
                            this._data = new Uint8Array(tensor.int32_data);
                            this._encoding = '|';
                            break;
                        case onnx.DataType.INT16:
                            this._data = new Int32Array(tensor.int32_data);
                            this._encoding = '|';
                            break;
                        case onnx.DataType.UINT16:
                            this._data = new Int32Array(tensor.int32_data);
                            this._encoding = '|';
                            break;
                        case onnx.DataType.INT32:
                            this._data = new Int32Array(tensor.int32_data);
                            this._encoding = '|';
                            break;
                        case onnx.DataType.UINT32:
                        case onnx.DataType.UINT64:
                            this._data = tensor.uint64_data;
                            this._encoding = '|';
                            break;
                        case onnx.DataType.INT64:
                            this._data = tensor.int64_data;
                            this._encoding = '|';
                            break;
                        case onnx.DataType.STRING:
                            this._data = tensor.string_data;
                            this._encoding = '|';
                            break;
                        case onnx.DataType.COMPLEX64:
                        case onnx.DataType.COMPLEX128:
                            break;
                        case onnx.DataType.FLOAT16:
                        case onnx.DataType.BFLOAT16:
                            if (tensor.int32_data && tensor.int32_data.length > 0) {
                                const array = tensor.int32_data;
                                const buffer = new Uint8Array(array.length << 1);
                                const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
                                for (let i = 0; i < array.length; i++) {
                                    view.setUint16(i << 1, array[i], true);
                                }
                                this._data = buffer;
                                this._encoding = '<';
                            }
                            break;
                        case onnx.DataType.FLOAT4E2M1:
                        case onnx.DataType.FLOAT8E4M3FN:
                        case onnx.DataType.FLOAT8E4M3FNUZ:
                        case onnx.DataType.FLOAT8E5M2:
                        case onnx.DataType.FLOAT8E5M2FNUZ:
                            if (tensor.int32_data && tensor.int32_data.length > 0) {
                                this._data = new Uint8Array(Array.from(tensor.int32_data));
                                this._encoding = '<';
                            }
                            break;
                        case onnx.DataType.UINT4:
                        case onnx.DataType.INT4:
                            if (tensor.int32_data && tensor.int32_data.length > 0) {
                                this._data = new Uint8Array(Array.from(tensor.int32_data));
                                this._encoding = '<';
                            }
                            break;
                        default:
                            throw new onnx.Error(`Unsupported tensor data type '${tensor.data_type}'.`);
                    }
                    if (this._data && (Array.isArray(this._data) || ArrayBuffer.isView(this._data)) && this._data.length === 0) {
                        this._data = undefined;
                    }
                    if (!this._data && tensor.raw_data && tensor.raw_data.length > 0) {
                        this._data = tensor.raw_data;
                        this._encoding = '<';
                    }
                    break;
                }
                case onnx.DataLocation.EXTERNAL: {
                    if (Array.isArray(tensor.external_data)) {
                        const data = new Map();
                        for (const entry of tensor.external_data) {
                            data.set(entry.key, entry.value);
                        }
                        if (data.has('location')) {
                            this._location = data.get('location').toString();
                            const location = context.location(this._location);
                            const offset = data.has('offset') ? parseInt(data.get('offset'), 10) : 0;
                            const length = data.has('length') ? parseInt(data.get('length'), 10) : -1;
                            this._request = { location, offset, length };
                            this._encoding = '<';
                        }
                    }
                    break;
                }
                default: {
                    break;
                }
            }
        }
    }

    peek() {
        return !this._request;
    }

    async read() {
        if (this._request) {
            const location = this._request.location;
            const offset = this._request.offset;
            const length = this._request.length;
            this._data = await location.read(offset, length);
            delete this._request;
        }
    }

    get name() {
        return this._name;
    }

    get category() {
        return this._category;
    }

    get encoding() {
        return this._encoding;
    }

    get location() {
        return this._location;
    }

    get type() {
        return this._type;
    }

    get indices() {
        return this._indices;
    }

    get values() {
        if (this._request) {
            throw new onnx.Error('Tensor data not loaded.');
        }
        switch (this.type.layout) {
            case 'sparse': {
                return this._values;
            }
            default: {
                if (!this._data || this._data instanceof Uint8Array) {
                    return this._data;
                }
                if (Array.isArray(this._data) || ArrayBuffer.isView(this._data)) {
                    return this._data;
                }
                return this._data.peek();
            }
        }
    }
};

enlight.TensorType = class {

    constructor(dataType, shape, layout, denotation) {
        this._dataType = dataType;
        this._shape = shape;
        this._layout = layout || null;
        this._denotation = denotation || null;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    get layout() {
        return this._layout;
    }

    get denotation() {
        return this._denotation;
    }

    toString() {
        return this.dataType + this._shape.toString();
    }
};

enlight.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions.map((dim) => typeof dim === 'bigint' ? dim.toNumber() : dim);
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length === 0) {
            return '';
        }
        return `[${this._dimensions.map((dim) => dim || Number.isInteger(dim) ? dim.toString() : '?').join(',')}]`;
    }
};

enlight.SequenceType = class {

    constructor(elementType, denotation) {
        this._elementType = elementType;
        this._denotation = denotation;
    }

    get elementType() {
        return this._elementType;
    }

    get dennotation() {
        return this._dennotation;
    }

    toString() {
        const elementType = this._elementType ? this._elementType.toString() : '';
        return `sequence<${elementType}>`;
    }
};

enlight.MapType = class {

    constructor(keyType, valueType, denotation) {
        this._keyType = keyType;
        this._valueType = valueType;
        this._denotation = denotation;
    }

    get keyType() {
        return this._keyType;
    }

    get valueType() {
        return this._valueType;
    }

    get denotation() {
        return this._denotation;
    }

    toString() {
        return `map<${this._keyType},${this._valueType}>`;
    }
};

enlight.OpaqueType = class {

    constructor(domain, name) {
        this._domain = domain;
        this._name = name;
    }

    toString() {
        const name = (this._domain ? (`${this._domain}.`) : '') + this._name;
        return `opaque<${name}>`;
    }
};

enlight.OptionalType = class {

    constructor(type) {
        this._type = type;
    }

    get type() {
        return this._type;
    }

    toString() {
        return `optional<${this._type}>`;
    }
};

enlight.Function = class {

    constructor(context, func) {
        this.type = 'function';
        this.name = func.name;
        this.module = func.domain;
        this.overload = func.overload || '';
        this.identifier = this.overload ? `${this.name}:${this.overload}` : this.name;
        this.description = func.doc_string;
        this.inputs = [];
        this.outputs = [];
        this.attributes = func.attribute.map((attribtue) => {
            return { name: attribtue };
        });
        context = new onnx.Context.Graph(context, func);
        func.input = func.input.map((input) => context.tensor(input));
        func.output = func.output.map((output) => context.tensor(output));
        context.push(func.node, func.input, func.output);
        this.nodes = context.pop();
        for (const input of func.input) {
            const value = context.value(input.name);
            if (!value.initializer) {
                this.inputs.push(new onnx.Argument(input.name, [value]));
            }
        }
        for (const output of func.output) {
            const value = context.value(output.name);
            if (!value.initializer) {
                this.outputs.push(new onnx.Argument(output.name, [value]));
            }
        }
    }
};

enlight.Context = class {};

enlight.Context.Model = class {

    constructor(metadata, locations, imageFormat, imports, graph, functions) {
        this._metadata = metadata;
        this._locations = locations;
        this._imageFormat = imageFormat;
        this._imports = imports;
        this._types = new Map();
        this._attributes = new Map();
        this._graph = null;
        this._functions = new Map();
        for (const func of functions || []) {
            const key = func.overload ? `${func.domain}:${func.name}:${func.overload}` : `${func.domain}:${func.name}`;
            func.initializer = [];
            func.uses = [];
            func.callers = new Set();
            this._functions.set(key, func);
        }
        if (graph) {
            if (this._functions.size > 0) { // #1208
                const queue = [graph].concat(Array.from(this._functions.values()));
                for (const graph of queue) {
                    const graphs = [graph];
                    while (graphs.length > 0) {
                        const graph = graphs.shift();
                        for (const node of graph.node) {
                            const key = node.overload ? `${node.domain}:${node.op_type}:${node.overload}` : `${node.domain}:${node.op_type}`;
                            if (this._functions.has(key)) {
                                this._functions.get(key).callers.add(graph);
                            }
                            for (const attribute of node.attribute) {
                                if (attribute.g) {
                                    graphs.push(attribute.g);
                                }
                                if (Array.isArray(attribute.graphs) && attribute.graphs.length > 0) {
                                    graphs.push(...attribute.graphs);
                                }
                            }
                        }
                    }
                }
                const visited = new Set();
                const graphs = new Set([graph]);
                while (graphs.size > 0) {
                    const graph = graphs.values().next().value;
                    graphs.delete(graph);
                    if (visited.has(graph)) {
                        continue;
                    }
                    if (graph.callers && !Array.from(graph.callers).every((caller) => visited.has(caller))) {
                        graphs.add(graph);
                        continue;
                    }
                    visited.add(graph);
                    graph.initializer = graph.initializer || [];
                    const initializers = new Map();
                    for (const initializer of graph.initializer) {
                        initializers.set(initializer.name, { uses: [], initializer, visible: true });
                    }
                    for (const node of graph.node) {
                        const key = node.overload ? `${node.domain}:${node.op_type}:${node.overload}` : `${node.domain}:${node.op_type}`;
                        if (this._functions.has(key)) {
                            this._functions.get(key).uses.push(node);
                        }
                        for (const input of node.input) {
                            if (initializers.has(input)) {
                                initializers.get(input).uses.push(node);
                            }
                        }
                        for (const attribute of node.attribute) {
                            if (attribute.g) {
                                graphs.add(attribute.g);
                            }
                            if (Array.isArray(attribute.graphs) && attribute.graphs.length > 0) {
                                for (const graph of attribute.graphs) {
                                    graphs.add(graph);
                                }
                            }
                        }
                    }
                    const queue = [];
                    for (const [name, entry] of initializers) {
                        if (entry.uses.length === 1) {
                            const [node] = entry.uses;
                            const key = node.overload ? `${node.domain}:${node.op_type}:${node.overload}` : `${node.domain}:${node.op_type}`;
                            if (this._functions.has(key)) {
                                const func = this._functions.get(key);
                                if (func.uses.length === 1 && func.callers.size === 1) {
                                    const index = node.input.indexOf(name);
                                    if (Array.isArray(func.input) && index < func.input.length && func.input[index] === name) {
                                        func.initializer.push(entry.initializer);
                                        graphs.add(func);
                                        queue.push([index, node]);
                                    }
                                }
                            }
                        }
                    }
                    queue.sort((a, b) => b[0] - a[0]);
                    for (const [index, node] of queue) {
                        node.input.splice(index, 1);
                    }
                }
            }
            this._graph = new enlight.Graph(this, graph);
        }
    }

    get imageFormat()  {
        return this._imageFormat;
    }

    get graph() {
        return this._graph;
    }

    location(name) {
        if (this._locations.has(name)) {
            return this._locations.get(name);
        }
        return null;
    }

    initializer(/* name */) {
        return null;
    }

    type(domain, name, overload) {
        const key = overload ? `${domain}:${name}:${overload}` : `${domain}:${name}`;
        if (!this._types.has(key)) {
            let value = null;
            if (this._functions.has(key)) {
                value = this._functions.get(key);
                if (value.domain !== undefined) {
                    value = new onnx.Function(this, value);
                    this._functions.set(key, value);
                }
            }
            if (!value) {
                value = this._metadata.type(domain, name, this._imports);
            }
            this._types.set(key, value);
        }
        return this._types.get(key);
    }

    attribute(domain, type, overload, name) {
        const key = overload ? `${domain}:${type}:${overload}::${name}` : `${domain}:${type}::${name}`;
        if (!this._attributes.has(key)) {
            this._attributes.set(key, null);
            const metadata = this.type(domain, type);
            if (metadata && Array.isArray(metadata.attributes) && metadata.attributes.length > 0) {
                for (const attribute of metadata.attributes) {
                    const name = attribute.name;
                    const key = overload ? `${domain}:${type}:${overload}::${name}` : `${domain}:${type}::${name}`;
                    this._attributes.set(key, attribute);
                }
            }
        }
        return this._attributes.get(key);
    }
};

enlight.Metadata = class {

    static async open(context) {
        if (!enlight.Metadata._metadata) {
            let data = null;
            try {
                data = await context.request('enlight-metadata.json');
            } catch {
                // continue regardless of error
            }
            enlight.Metadata._metadata = new enlight.Metadata(data);
        }
        return enlight.Metadata._metadata;
    }

    constructor(data) {
        this._types = new Map();
        if (data) {
            const types = JSON.parse(data);
            for (const type of types) {
                if (!this._types.has(type.module)) {
                    this._types.set(type.module, new Map());
                }
                const types = this._types.get(type.module);
                if (!types.has(type.name)) {
                    types.set(type.name, []);
                }
                types.get(type.name).push(type);
            }
        }
    }

    type(domain, name, imports) {
        domain = domain || 'ai.onnx';
        let current = null;
        if (this._types.has(domain)) {
            const types = this._types.get(domain);
            if (types.has(name)) {
                for (const type of types.get(name)) {
                    const matchVersion = current ? current.version : -1;
                    const importVersion = imports.get(type.module) || 0;
                    if (importVersion >= type.version && matchVersion < type.version) {
                        current = type;
                    }
                }
            }
        }
        return current;
    }
};

enlight.Inference = class {

    constructor(nodes) {
        this._outputs = new Map();
        for (const node of nodes) {
            for (const output of node.output) {
                this._outputs.set(output.name, node);
            }
        }
    }

    infer(output) {
        if (this._outputs.has(output)) {
            let hasInputShapes = true;
            const node = this._outputs.get(output);
            for (const input of node.input) {
                if (!input.type) {
                    this.infer(input);
                    if (!input.type) {
                        hasInputShapes = false;
                        break;
                    }
                }
            }
            if (hasInputShapes) {
                // continue
            }
        }
    }
};

enlight.DataLocation = {
    DEFAULT: 0,
    EXTERNAL: 1
};

enlight.DataType = {
    UNDEFINED: 0,
    FLOAT: 1,
    UINT8: 2,
    INT8: 3,
    UINT16: 4,
    INT16: 5,
    INT32: 6,
    INT64: 7,
    STRING: 8,
    BOOL: 9,
    FLOAT16: 10,
    DOUBLE: 11,
    UINT32: 12,
    UINT64: 13,
    COMPLEX64: 14,
    COMPLEX128: 15,
    BFLOAT16: 16,
    FLOAT8E4M3FN: 17,
    FLOAT8E4M3FNUZ: 18,
    FLOAT8E5M2: 19,
    FLOAT8E5M2FNUZ: 20,
    UINT4: 21,
    INT4: 22,
    FLOAT4E2M1: 23
};

enlight.AttributeType = {
    UNDEFINED: 0,
    FLOAT: 1,
    INT: 2,
    STRING: 3,
    TENSOR: 4,
    GRAPH: 5,
    FLOATS: 6,
    INTS: 7,
    STRINGS: 8,
    TENSORS: 9,
    GRAPHS: 10,
    SPARSE_TENSOR: 11,
    SPARSE_TENSORS: 12,
    TYPE_PROTO: 13,
    TYPE_PROTOS: 14
};

enlight.Context.Graph = class {

    constructor(context, graph) {
        this._context = context;
        this._dataTypes = new Map(Object.entries(enlight.schema.DataType).map(([name, value]) => [value, name.toLowerCase()]));
        // this._dataTypes.set(enlight.schema.DataType.UNDEFINED, 'undefined');
        // this._dataTypes.set(enlight.schema.DataType.BOOL, 'boolean');
        // this._dataTypes.set(enlight.schema.DataType.FLOAT, 'float32');
        // this._dataTypes.set(enlight.schema.DataType.DOUBLE, 'float64');
        this._graphs = new Map();
        this._initializers = new Map();
        this._tensors = new Map();
        this._values = new Map();
        this._groups = new Map();
        this._nodes = [];

        const params = {};
        // generate parameters
        let paramIdx = 0;
        for (let j = 0; j < graph.layersLength(); j++) {
            const base = enlight.Context.Node.getBase(graph.layers(j));
            for (let i = 0 ; i < base.outputSlotsLength() ; i++) {
                const slot = base.outputSlots(i);
                const key = enlight.Context.Parameter.makeKey(base.index(), i);
                const name = paramIdx.toString();

                let stats = null;
                let threshold = null;

                if (slot.statisticsEnabled()) {
                    stats = [slot.min(), slot.max(), slot.mean(), slot.std()];
                }

                if (slot.thresholdEnabled()) {
                    threshold = slot.threshold();
                }

                const args = [new enlight.Context.Argument(name, slot.tensorInfo(), null, stats, threshold)];
                params[key] = new enlight.Context.Parameter(name, name, args);
                paramIdx++;
            }
        }

        if (Array.isArray(graph.initializer)) {
            for (const initializer of graph.initializer) {
                const tensor = new enlight.Tensor(this, initializer, 'Initializer');
                this._initializers.set(initializer.name, tensor);
            }
        }
        if (Array.isArray(graph.sparse_initializer)) {
            for (const sparse_initializer of graph.sparse_initializer) {
                const tensor = new enlight.Tensor(this, sparse_initializer, 'Initializer');
                this._initializers.set(sparse_initializer.values.name, tensor);
            }
        }
        for (const node of graph.node) {
            node.input = node.input.map((name) => this.tensor(name));
            node.output = node.output.map((name) => this.tensor(name));
            node.param = {};
            if (Array.isArray(node.attribute)) {
                for (const attribute of node.attribute) {
                    if (attribute.type) {
                        continue;
                    }
                    if (Array.isArray(attribute.ints) && attribute.ints.length > 0) {
                        attribute.type = enlight.AttributeType.INTS;
                    } else if (Array.isArray(attribute.floats) && attribute.floats.length > 0) {
                        attribute.type = enlight.AttributeType.FLOATS;
                    } else if (Array.isArray(attribute.strings) && attribute.strings.length > 0) {
                        attribute.type = enlight.AttributeType.STRINGS;
                    } else if (Array.isArray(attribute.graphs) && attribute.graphs.length > 0) {
                        attribute.type = enlight.AttributeType.GRAPHS;
                    } else if (Array.isArray(attribute.s) && attribute.s.length > 0) {
                        attribute.type = enlight.AttributeType.STRING;
                    } else if (attribute.f !== undefined) {
                        attribute.type = enlight.AttributeType.FLOAT;
                    } else if (attribute.i !== undefined) {
                        attribute.type = enlight.AttributeType.INT;
                    } else if (attribute.t !== undefined) {
                        attribute.type = enlight.AttributeType.TENSOR;
                    } else if (attribute.g !== undefined) {
                        attribute.type = enlight.AttributeType.GRAPH;
                    } else if (attribute.sparse_tensor) {
                        attribute.type = enlight.AttributeType.SPARSE_TENSOR;
                    } else {
                        attribute.type = enlight.AttributeType.UNDEFINED;
                    }
                }
            }
        }
    }

    type(domain, name, overload) {
        return this._context.type(domain, name, overload);
    }

    attribute(domain, type, overload, name) {
        return this._context.attribute(domain, type, overload, name);
    }

    graph(value) {
        if (!this._graphs.has(value)) {
            this._graphs.set(value, new enlight.Graph(this, value));
        }
        return this._graphs.get(value);
    }

    initializer(name) {
        if (this._initializers.has(name)) {
            return this._initializers.get(name);
        }
        return this._context.initializer(name);
    }

    tensor(name) {
        if (!this._tensors.has(name)) {
            this._tensors.set(name, { name, initializer: this.initializer(name) });
        }
        return this._tensors.get(name);
    }

    location(name) {
        return this._context.location(name);
    }

    group(name) {
        if (!this._groups.has(name)) {
            const path = name.split('/');
            if (path.length > 1) {
                path.pop();
                return this.group(path.join('/'));
            }
            this._groups.set(name, new Map([['', []]]));
        }
        return this._groups.get(name);
    }

    value(name) {
        if (!this._values.has(name)) {
            const tensor = this.tensor(name);
            const type = tensor.initializer ? tensor.initializer.type : tensor.type || null;
            this._values.set(name, new enlight.Value(name, type, tensor.initializer, tensor.annotation, tensor.description));
        }
        return this._values.get(name);
    }

    createType(type) {
        if (!type) {
            return null;
        }
        let denotation = '';
        switch (type.denotation) {
            case undefined:
            case null:
            case '':
                break;
            case 'TENSOR':
                denotation = 'Tensor';
                break;
            case 'IMAGE':
                denotation = `Image${this._context.imageFormat ? `(${this._context.imageFormat.join(',')})` : ''}`;
                break;
            case 'AUDIO':
                denotation = 'Audio';
                break;
            case 'TEXT':
                denotation = 'Text';
                break;
            default:
                throw new enlight.Error(`Unsupported tensor type denotation '${type.denotation}'.`);
        }
        if (type.tensor_type) {
            const tensor_type = type.tensor_type;
            const shape = tensor_type.shape && tensor_type.shape.dim ? tensor_type.shape.dim.map((dim) => dim.dim_param ? dim.dim_param : dim.dim_value || null) : [];
            return this.createTensorType(tensor_type.elem_type, shape, null, denotation);
        } else if (type.sparse_tensor_type) {
            type = type.sparse_tensor_type;
            const shape = type.shape && type.shape.dim ? type.shape.dim.map((dim) => dim.dim_param ? dim.dim_param : dim.dim_value || null) : [];
            return this.createTensorType(type.elem_type, shape, 'sparse', denotation);
        } else if (type.map_type) {
            const keyType = this.createDataType(type.map_type.key_type);
            const valueType = this.createType(type.map_type.value_type);
            return new enlight.MapType(keyType, valueType, denotation);
        } else if (type.sequence_type) {
            return new enlight.SequenceType(this.createType(type.sequence_type.elem_type), denotation);
        } else if (type.opaque_type) {
            return new enlight.OpaqueType(type.opaque_type.domain, type.opaque_type.name);
        } else if (type.optional_type) {
            return new enlight.OptionalType(this.createType(type.optional_type.elem_type), denotation);
        } else if (Object.keys(type).length === 0) {
            return null;
        }
        throw new enlight.Error(`Unsupported tensor type '${JSON.stringify(type)}'.`);
    }

    createTensorType(dataType, shape, layout, denotation) {
        dataType = this.createDataType(dataType);
        return new onnx.TensorType(dataType, new onnx.TensorShape(shape), layout, denotation);
    }

    createDataType(value) {
        if (!Number.isInteger(value)) {
            if (typeof value === 'bigint') {
                value = value.toNumber();
            } else if (value && typeof value === 'string' && onnx.DataType[value.toUpperCase()] !== undefined) {
                value = onnx.DataType[value.toUpperCase()];
            } else {
                throw new onnx.Error(`Unsupported data type '${JSON.stringify(value)}'.`);
            }
        }
        if (this._dataTypes.has(value)) {
            return this._dataTypes.get(value);
        }
        throw new onnx.Error(`Unsupported data type '${JSON.stringify(value)}'.`);
    }

    createLocation(value) {
        switch (value) {
            case undefined:
            case onnx.DataLocation.DEFAULT: return '';
            case onnx.DataLocation.EXTERNAL: return 'external';
            default: return 'undefined';
        }
    }

    decodeText(value) {
        if (typeof value === 'string') {
            return value;
        }
        this._decoder = this._decoder || new TextDecoder('utf-8');
        return this._decoder.decode(value);
    }

    push(nodes, inputs, outputs) {
        const inputMap = new Map();
        const outputMap = new Map();
        for (const node of nodes) {
            for (const input of node.input) {
                inputMap.set(input.name, (inputMap.get(input) || 0) + 1);
            }
            for (const output of node.output) {
                outputMap.set(output.name, (outputMap.get(output) || 0) + 1);
            }
        }
        inputs.every((input) => inputMap.delete(input.name));
        outputs.every((output) => outputMap.delete(output.name));
        nodes = nodes.filter((node) => {
            const constant = node &&
                node.op_type === 'Constant' &&
                node.attribute.length === 1 && node.attribute[0] &&
                node.input.length === 0 &&
                node.output.length === 1 && node.output[0] && inputMap.get(node.output[0].name) === 1 && outputMap.get(node.output[0].name) === 1;
            const attribute = constant ? node.attribute[0] : null;
            if (attribute && attribute.name === 'value' && attribute.type === onnx.AttributeType.TENSOR && attribute.t) {
                const tensor = this.tensor(node.output[0].name);
                tensor.initializer = new onnx.Tensor(this, attribute.t, 'Constant');
                return false;
            } else if (attribute && attribute.name === 'sparse_value' && attribute.type === onnx.AttributeType.SPARSE_TENSOR && attribute.sparse_tensor) {
                const tensor = this.tensor(node.output[0].name);
                tensor.initializer = new onnx.Tensor(this, attribute.sparse_tensor, 'Constant');
                return false;
            }
            return true;
        });
        for (let node of nodes) {
            node = new onnx.Node(this, node);
            this._nodes.push(node);

            // const path = (node.name || '').split('/');
            // path.pop();
            // this.group(path.join('/')).get('').push(node);
        }
    }

    pop() {
        /*
        const nodes = [];
        for (const [name, value] of this._groups) {
            if (name === '') {
                for (const node of value.get('')) {
                    nodes.push(node);
                }
                continue;
            }
            nodes.push(new onnx.Group(name, value));
        }
        return nodes;
        */
        return this._nodes;
    }
};

enlight.Context.Node = class {

    constructor(layer, params, fused) {
        const op_type = enlight.schema.LayerName[layer.layerType()].replace(/Layer$/, '');

        // this.name = '';
        const outputs = [];
        const inputs = [];
        this._category = '';
        this._group_idc = null;
        this._position_in_group = -1;

        let base = null;

        if (!fused) {
            base = enlight.Context.Node.getBase(layer);
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

                inputs.push(params[enlight.Context.Parameter.makeKey(srcLayerIdx, srcOutputIdx)]);
            }

            for (let j = 0; j < base.outputSlotsLength(); j++) {
                outputs.push(params[enlight.Context.Parameter.makeKey(base.index(), j)]);
            }

            for (let j = 0; j < layer.fusedLayersLength(); j++) {
                const fusedLayer = layer.fusedLayers(j);
                chain.push(new enlight.Context.Node(fusedLayer, params, true));
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

enlight.Context.Tensor = class {

    constructor(tensorInfo, tensor) {
        this._name = '';
        this._type = new enlight.Context.TensorType(tensorInfo);
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

enlight.Context.TensorType = class {

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

enlight.Context.TensorShape = class {

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


enlight.Context.Parameter = class {

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

enlight.Context.Argument = class {

    constructor(id, tensorInfo, initializer, stats, threshold) {
        let info = initializer ? initializer.info() : tensorInfo;

        this._id = id;
        this._type = new enlight.Context.TensorType(info);
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

enlight.FlatReader = class {

    static open(context) {
        const identifier = context.identifier.toLowerCase();
        if (identifier.endsWith('.enlight')) {
            return new enlight.FlatReader(context, identifier);
        }
        return null;
    }

    constructor(context, identifier) {
        this.name = 'enlight.flat.data';
        this.context = context;
        this.identifier = identifier;
        this.locations = new Map();
        this.locations.set(identifier, context.stream);
    }

    async read() {
        const stream = this.locations.get(this.identifier);
        const buffer = stream._buffer;
        const byteBuffer = new flatbuffers.ByteBuffer(buffer);
        const schema = new enlight_schema.Network();
        this.model = schema.getRootAsNetwork(byteBuffer);
        this.format = 'Enlight FlatBuffer';
    }
};

enlight.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Enlight model.';
    }
};

export const ModelFactory = enlight.ModelFactory;
