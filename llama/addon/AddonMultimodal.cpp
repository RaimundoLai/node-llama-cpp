#include "AddonMultimodal.h"
#include <stdexcept>
#include <cmath>
#include <limits>
#include <memory>

namespace {
bool getBufferData(const Napi::Value& value, const uint8_t*& data, size_t& length) {
    if (value.IsBuffer()) {
        auto buffer = value.As<Napi::Buffer<uint8_t>>();
        data = buffer.Data();
        length = buffer.Length();
        return true;
    }

    if (!value.IsTypedArray()) {
        return false;
    }

    auto typedArray = value.As<Napi::TypedArray>();
    if (typedArray.TypedArrayType() != napi_uint8_array) {
        return false;
    }

    auto uint8Array = value.As<Napi::Uint8Array>();
    data = uint8Array.Data();
    length = uint8Array.ByteLength();
    return true;
}

// Pooled embeddings must see the complete input in one logical batch. Evaluating
// chunks separately resets the pooling output and breaks bidirectional attention.
int32_t evalPooledChunks(mtmd_context* multimodal, llama_context* context,
        const mtmd_input_chunks* chunks, llama_seq_id sequenceId, llama_pos firstPosition,
        size_t tokensCount, llama_pos* newPast) {
    if (tokensCount == 0) {
        *newPast = firstPosition;
        return 0;
    }
    if (tokensCount > llama_n_batch(context) || tokensCount > llama_n_ubatch(context)) {
        throw std::runtime_error("Multimodal embeddings require batchSize to fit the complete input. Increase batchSize or shorten the media.");
    }
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
        llama_batch_ext_init(context), llama_batch_ext_free);
    if (!batch) throw std::runtime_error("Failed to initialize embedding batch");
    const size_t dimensions = llama_model_n_embd_inp(llama_get_model(context));
    const bool mrope = mtmd_decode_use_mrope(multimodal);
    llama_pos position = firstPosition;
    for (size_t index = 0; index < mtmd_input_chunks_size(chunks); index++) {
        const auto* chunk = mtmd_input_chunks_get(chunks, index);
        const auto type = mtmd_input_chunk_get_type(chunk);
        const size_t count = mtmd_input_chunk_get_n_tokens(chunk);
        const float* encoded = nullptr;
        const llama_token* tokens = nullptr;
        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t textCount;
            tokens = mtmd_input_chunk_get_tokens_text(chunk, &textCount);
        } else {
            const int32_t status = mtmd_encode_chunk(multimodal, chunk);
            if (status != 0) return status;
            encoded = mtmd_get_output_embd(multimodal);
            if (!encoded) throw std::runtime_error("Failed to encode media for embedding");
        }
        for (size_t offset = 0; offset < count; offset++) {
            const int32_t row = tokens
                ? llama_batch_ext_add_token(batch.get(), sequenceId, tokens[offset])
                : llama_batch_ext_add_embd(batch.get(), sequenceId, {encoded + offset * dimensions, 1, dimensions});
            if (row < 0) throw std::runtime_error("Failed to add media embedding row to batch");
            llama_pos positions[4] = {position + static_cast<llama_pos>(offset), 0, 0, 0};
            if (mrope && encoded) {
                if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                    const auto relative = mtmd_image_tokens_get_decoder_pos(
                        mtmd_input_chunk_get_tokens_image(chunk), position, offset);
                    positions[0] = relative.t;
                    positions[1] = relative.y;
                    positions[2] = relative.x;
                    positions[3] = relative.z;
                } else {
                    positions[1] = positions[0];
                    positions[2] = positions[0];
                    positions[3] = static_cast<llama_pos>(offset);
                }
            }
            if (!llama_batch_ext_set_pos(batch.get(), row, positions) ||
                !llama_batch_ext_set_output_embd(batch.get(), row, true)) {
                throw std::runtime_error("Failed to configure media embedding row");
            }
        }
        position += mtmd_input_chunk_get_n_pos(chunk);
    }
    const int32_t status = llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    *newPast = position;
    return status;
}
}

class AddonMultimodalInitWorker : public Napi::AsyncWorker {
    public:
        AddonMultimodal* addonMultimodal;
        Napi::Promise::Deferred deferred;

        AddonMultimodalInitWorker(Napi::Env env, AddonMultimodal* addonMultimodal)
            : Napi::AsyncWorker(env, "AddonMultimodalInitWorker"),
              addonMultimodal(addonMultimodal),
              deferred(Napi::Promise::Deferred::New(env)) {}

        void Execute() override {
            try {
                addonMultimodal->mtmd_ctx = mtmd_init_from_file(
                    addonMultimodal->mmprojPath.c_str(),
                    addonMultimodal->model->model,
                    addonMultimodal->params
                );

                if (addonMultimodal->mtmd_ctx == nullptr) {
                    SetError("Failed to load multimodal projector from " + addonMultimodal->mmprojPath);
                }
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch (...) {
                SetError("Unknown error when loading multimodal projector");
            }
        }

        void OnOK() override {
            deferred.Resolve(Env().Undefined());
        }

        void OnError(const Napi::Error& err) override {
            deferred.Reject(err.Value());
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }
};

class AddonMultimodalEvalWorker : public Napi::AsyncWorker {
    public:
        AddonMultimodal* addonMultimodal;
        AddonContext* context;
        Napi::Reference<Napi::Object> contextRef;
        std::vector<Napi::Reference<Napi::Object>> bitmapRefs;
        std::vector<const mtmd_bitmap*> rawBitmaps;
        int32_t sequenceId;
        llama_pos firstTokenContextIndex;
        std::string prompt;
        bool logitsLast;
        bool addSpecial;
        bool parseSpecial;
        int32_t nBatch;

        llama_pos resNewPast = 0;
        size_t totalTokens = 0;
        Napi::Promise::Deferred deferred;

        AddonMultimodalEvalWorker(
            Napi::Env env,
            AddonMultimodal* addonMultimodal,
            AddonContext* context,
            Napi::Reference<Napi::Object> contextRef,
            std::vector<Napi::Reference<Napi::Object>> bitmapRefs,
            std::vector<const mtmd_bitmap*> rawBitmaps,
            int32_t sequenceId,
            llama_pos firstTokenContextIndex,
            std::string prompt,
            bool logitsLast,
            bool addSpecial,
            bool parseSpecial,
            int32_t nBatch
        ) : Napi::AsyncWorker(env, "AddonMultimodalEvalWorker"),
            addonMultimodal(addonMultimodal),
            context(context),
            contextRef(std::move(contextRef)),
            bitmapRefs(std::move(bitmapRefs)),
            rawBitmaps(std::move(rawBitmaps)),
            sequenceId(sequenceId),
            firstTokenContextIndex(firstTokenContextIndex),
            prompt(std::move(prompt)),
            logitsLast(logitsLast),
            addSpecial(addSpecial),
            parseSpecial(parseSpecial),
            nBatch(nBatch),
            deferred(Napi::Promise::Deferred::New(env)) {}

        void Execute() override {
            try {
                if (addonMultimodal->disposed || addonMultimodal->mtmd_ctx == nullptr) {
                    SetError("Multimodal context is disposed");
                    return;
                }
                if (context->disposed || context->ctx == nullptr) {
                    SetError("Context is disposed");
                    return;
                }

                mtmd_input_chunks* chunks = mtmd_input_chunks_init();
                if (chunks == nullptr) {
                    SetError("Failed to initialize mtmd input chunks");
                    return;
                }

                mtmd_input_text inputText = {};
                inputText.text = prompt.c_str();
                inputText.text_len = prompt.size();
                inputText.add_special = addSpecial;
                inputText.parse_special = parseSpecial;

                int32_t tokenRes = mtmd_tokenize(
                    addonMultimodal->mtmd_ctx,
                    chunks,
                    &inputText,
                    rawBitmaps.data(),
                    rawBitmaps.size()
                );

                if (tokenRes != 0) {
                    mtmd_input_chunks_free(chunks);
                    if (tokenRes == 1) {
                        SetError("Number of bitmaps (" + std::to_string(rawBitmaps.size()) +
                                 ") does not match the number of media markers in prompt");
                    } else {
                        SetError("Failed to tokenize multimodal prompt (code: " + std::to_string(tokenRes) + ")");
                    }
                    return;
                }

                totalTokens = mtmd_helper_get_n_tokens(chunks);
                // Positions can differ from tokens for models using multidimensional RoPE.
                const auto positions = mtmd_helper_get_n_pos(chunks);
                if (firstTokenContextIndex < 0 ||
                    static_cast<uint64_t>(firstTokenContextIndex) + positions > llama_n_ctx_seq(context->ctx)) {
                    mtmd_input_chunks_free(chunks);
                    SetError("Multimodal input exceeds the sequence context size. Use shorter media or a larger context.");
                    return;
                }
                llama_pos newPast = firstTokenContextIndex;
                int32_t evalRes;
                try {
                    evalRes = llama_pooling_type(context->ctx) != LLAMA_POOLING_TYPE_NONE
                        ? evalPooledChunks(addonMultimodal->mtmd_ctx, context->ctx, chunks,
                            sequenceId, firstTokenContextIndex, totalTokens, &newPast)
                        : mtmd_helper_eval_chunks(addonMultimodal->mtmd_ctx, context->ctx, chunks,
                            firstTokenContextIndex, sequenceId, nBatch, logitsLast, &newPast);
                } catch (...) {
                    mtmd_input_chunks_free(chunks);
                    throw;
                }
                mtmd_input_chunks_free(chunks);

                if (evalRes != 0) {
                    SetError("Failed to evaluate multimodal chunks (code: " + std::to_string(evalRes) + ")");
                    return;
                }

                resNewPast = newPast;
            } catch (const std::exception& e) {
                SetError(e.what());
            } catch (...) {
                SetError("Unknown error when evaluating multimodal chunks");
            }
        }

        void OnOK() override {
            Napi::Object result = Napi::Object::New(Env());
            result.Set("newPast", Napi::Number::New(Env(), resNewPast));
            result.Set("tokensCount", Napi::Number::New(Env(), totalTokens));
            deferred.Resolve(result);
        }

        void OnError(const Napi::Error& err) override {
            deferred.Reject(err.Value());
        }

        Napi::Promise GetPromise() {
            return deferred.Promise();
        }
};

AddonMultimodal::AddonMultimodal(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<AddonMultimodal>(info) {
    if (info.Length() < 2 || !info[0].IsObject() || !info[1].IsObject()) {
        Napi::TypeError::New(info.Env(), "Expected model and options objects as arguments").ThrowAsJavaScriptException();
        return;
    }

    model = Napi::ObjectWrap<AddonModel>::Unwrap(info[0].As<Napi::Object>());
    modelRef = Napi::Persistent(info[0].As<Napi::Object>());
    hasModelRef = true;

    auto options = info[1].As<Napi::Object>();
    if (!options.Has("modelPath") || !options.Get("modelPath").IsString()) {
        Napi::TypeError::New(info.Env(), "Expected modelPath in options").ThrowAsJavaScriptException();
        return;
    }

    mmprojPath = options.Get("modelPath").As<Napi::String>().Utf8Value();
    params = mtmd_context_params_default();

    if (options.Has("useGpu")) {
        params.use_gpu = options.Get("useGpu").As<Napi::Boolean>().Value();
    }
    if (options.Has("printTimings")) {
        params.print_timings = options.Get("printTimings").As<Napi::Boolean>().Value();
    }
    if (options.Has("nThreads")) {
        params.n_threads = options.Get("nThreads").As<Napi::Number>().Int32Value();
    }
    if (options.Has("warmup")) {
        params.warmup = options.Get("warmup").As<Napi::Boolean>().Value();
    }
    if (options.Has("imageMinTokens")) {
        params.image_min_tokens = options.Get("imageMinTokens").As<Napi::Number>().Int32Value();
    }
    if (options.Has("imageMaxTokens")) {
        params.image_max_tokens = options.Get("imageMaxTokens").As<Napi::Number>().Int32Value();
    }
    if (options.Has("mediaMarker")) {
        mediaMarker = options.Get("mediaMarker").As<Napi::String>().Utf8Value();
        params.media_marker = mediaMarker.c_str();
    }
}

AddonMultimodal::~AddonMultimodal() {
    disposeMT();
}

void AddonMultimodal::disposeMemory() {
    mtmd_context* currentCtx = nullptr;
    {
        std::lock_guard<std::mutex> lock(disposeMutex);
        if (memoryDisposed) {
            return;
        }
        memoryDisposed = true;
        currentCtx = mtmd_ctx;
        mtmd_ctx = nullptr;
    }

    if (currentCtx != nullptr) {
        mtmd_free(currentCtx);
    }
}

void AddonMultimodal::disposeMT() {
    {
        std::lock_guard<std::mutex> lock(disposeMutex);
        if (disposed) {
            return;
        }
        disposed = true;
    }

    disposeMemory();

    if (hasModelRef) {
        hasModelRef = false;
        modelRef.Reset();
    }
}

Napi::Value AddonMultimodal::Init(const Napi::CallbackInfo& info) {
    if (disposed) {
        Napi::Error::New(info.Env(), "Multimodal context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    AddonMultimodalInitWorker* worker = new AddonMultimodalInitWorker(info.Env(), this);
    worker->Queue();
    return worker->GetPromise();
}

Napi::Value AddonMultimodal::Dispose(const Napi::CallbackInfo& info) {
    disposeMT();
    return info.Env().Undefined();
}

Napi::Value AddonMultimodal::GetDisposed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), disposed);
}

Napi::Value AddonMultimodal::SupportVision(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), (mtmd_ctx != nullptr && !disposed) ? mtmd_support_vision(mtmd_ctx) : false);
}

Napi::Value AddonMultimodal::SupportAudio(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), (mtmd_ctx != nullptr && !disposed) ? mtmd_support_audio(mtmd_ctx) : false);
}

Napi::Value AddonMultimodal::SupportVideo(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), (mtmd_ctx != nullptr && !disposed) ? mtmd_helper_support_video(mtmd_ctx) : false);
}

Napi::Value AddonMultimodal::GetDefaultMarker(const Napi::CallbackInfo& info) {
    const char* marker = (mtmd_ctx != nullptr && !disposed)
        ? mtmd_get_marker(mtmd_ctx)
        : mtmd_default_marker();
    return Napi::String::New(info.Env(), marker);
}

Napi::Value AddonMultimodal::CreateBitmapFromFile(const Napi::CallbackInfo& info) {
    if (disposed || mtmd_ctx == nullptr) {
        Napi::Error::New(info.Env(), "Multimodal context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(info.Env(), "Expected string file path as first argument").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    std::string filePath = info[0].As<Napi::String>().Utf8Value();
    auto initOptions = mtmd_helper_init_opt_default();
    if (info.Length() > 1 && info[1].IsObject() && info[1].As<Napi::Object>().Has("videoFps")) {
        Napi::Value fpsValue = info[1].As<Napi::Object>().Get("videoFps");
        if (!fpsValue.IsNumber()) {
            Napi::TypeError::New(info.Env(), "videoFps must be a positive number").ThrowAsJavaScriptException();
            return info.Env().Undefined();
        }
        double fps = fpsValue.As<Napi::Number>().DoubleValue();
        if (!std::isfinite(fps) || fps <= 0) {
            Napi::RangeError::New(info.Env(), "videoFps must be a positive finite number").ThrowAsJavaScriptException();
            return info.Env().Undefined();
        }
        initOptions.video_params.fps_target = static_cast<float>(fps);
    }

    auto bitmapWrapper = mtmd_helper_bitmap_init_from_file(mtmd_ctx, filePath.c_str(), false, initOptions);
    if (bitmapWrapper.bitmap == nullptr) {
        Napi::Error::New(info.Env(), "Failed to create bitmap from file: " + filePath).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    return AddonBitmap::NewInstance(info.Env(), bitmapWrapper.bitmap, bitmapWrapper.video_ctx);
}

Napi::Value AddonMultimodal::CreateBitmapFromBuffer(const Napi::CallbackInfo& info) {
    if (disposed || mtmd_ctx == nullptr) {
        Napi::Error::New(info.Env(), "Multimodal context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    if (info.Length() < 1) {
        Napi::TypeError::New(info.Env(), "Expected Buffer or Uint8Array as first argument").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    const uint8_t* data = nullptr;
    size_t length = 0;
    if (!getBufferData(info[0], data, length)) {
        Napi::TypeError::New(info.Env(), "Expected Buffer or Uint8Array as first argument").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    auto initOptions = mtmd_helper_init_opt_default();
    if (info.Length() > 1 && info[1].IsObject() && info[1].As<Napi::Object>().Has("videoFps")) {
        Napi::Value fpsValue = info[1].As<Napi::Object>().Get("videoFps");
        if (!fpsValue.IsNumber()) {
            Napi::TypeError::New(info.Env(), "videoFps must be a positive number").ThrowAsJavaScriptException();
            return info.Env().Undefined();
        }
        double fps = fpsValue.As<Napi::Number>().DoubleValue();
        if (!std::isfinite(fps) || fps <= 0) {
            Napi::RangeError::New(info.Env(), "videoFps must be a positive finite number").ThrowAsJavaScriptException();
            return info.Env().Undefined();
        }
        initOptions.video_params.fps_target = static_cast<float>(fps);
    }

    auto bitmapWrapper = mtmd_helper_bitmap_init_from_buf(
        mtmd_ctx,
        data,
        length,
        false,
        initOptions
    );
    if (bitmapWrapper.bitmap == nullptr) {
        Napi::Error::New(info.Env(), "Failed to create bitmap from buffer").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    return AddonBitmap::NewInstance(info.Env(), bitmapWrapper.bitmap, bitmapWrapper.video_ctx);
}

Napi::Value AddonMultimodal::EvalChunks(const Napi::CallbackInfo& info) {
    if (disposed || mtmd_ctx == nullptr) {
        Napi::Error::New(info.Env(), "Multimodal context is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    if (info.Length() < 6 || !info[0].IsObject() || !info[1].IsNumber() || !info[2].IsNumber() || !info[3].IsString() ||
        !info[4].IsArray() || !info[5].IsBoolean()) {
        Napi::TypeError::New(info.Env(), "Invalid arguments to evalChunks").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    AddonContext* context = Napi::ObjectWrap<AddonContext>::Unwrap(info[0].As<Napi::Object>());
    Napi::Reference<Napi::Object> contextRef = Napi::Persistent(info[0].As<Napi::Object>());
    int32_t sequenceId = info[1].As<Napi::Number>().Int32Value();
    llama_pos firstTokenContextIndex = info[2].As<Napi::Number>().Int32Value();
    std::string prompt = info[3].As<Napi::String>().Utf8Value();
    Napi::Array bitmapsArray = info[4].As<Napi::Array>();
    bool logitsLast = info[5].As<Napi::Boolean>().Value();

    bool addSpecial = true;
    bool parseSpecial = true;
    int32_t nBatch = context->context_params.n_batch;

    if (info.Length() > 6 && info[6].IsObject()) {
        auto evalOptions = info[6].As<Napi::Object>();
        if (evalOptions.Has("addSpecial")) {
            addSpecial = evalOptions.Get("addSpecial").As<Napi::Boolean>().Value();
        }
        if (evalOptions.Has("parseSpecial")) {
            parseSpecial = evalOptions.Get("parseSpecial").As<Napi::Boolean>().Value();
        }
        if (evalOptions.Has("nBatch")) {
            Napi::Value nBatchValue = evalOptions.Get("nBatch");
            if (!nBatchValue.IsNumber()) {
                Napi::TypeError::New(info.Env(), "nBatch must be a positive integer").ThrowAsJavaScriptException();
                return info.Env().Undefined();
            }

            double nBatchNumber = nBatchValue.As<Napi::Number>().DoubleValue();
            if (!std::isfinite(nBatchNumber) || nBatchNumber <= 0 || std::floor(nBatchNumber) != nBatchNumber ||
                nBatchNumber > static_cast<double>(std::numeric_limits<int32_t>::max())) {
                Napi::RangeError::New(info.Env(), "nBatch must be a positive integer").ThrowAsJavaScriptException();
                return info.Env().Undefined();
            }

            nBatch = static_cast<int32_t>(nBatchNumber);
        }
    }

    if (nBatch <= 0) {
        Napi::RangeError::New(info.Env(), "nBatch must be a positive integer").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    std::vector<Napi::Reference<Napi::Object>> bitmapRefs;
    std::vector<const mtmd_bitmap*> rawBitmaps;
    bitmapRefs.reserve(bitmapsArray.Length());
    rawBitmaps.reserve(bitmapsArray.Length());

    for (uint32_t i = 0; i < bitmapsArray.Length(); i++) {
        Napi::Value item = bitmapsArray.Get(i);
        if (!item.IsObject()) {
            Napi::TypeError::New(info.Env(), "Every item in bitmaps array must be an AddonBitmap").ThrowAsJavaScriptException();
            return info.Env().Undefined();
        }
        AddonBitmap* bmp = Napi::ObjectWrap<AddonBitmap>::Unwrap(item.As<Napi::Object>());
        if (bmp->disposed || bmp->bitmap == nullptr) {
            Napi::Error::New(info.Env(), "Bitmap is disposed").ThrowAsJavaScriptException();
            return info.Env().Undefined();
        }
        bitmapRefs.push_back(Napi::Persistent(item.As<Napi::Object>()));
        rawBitmaps.push_back(bmp->bitmap);
    }

    AddonMultimodalEvalWorker* worker = new AddonMultimodalEvalWorker(
        info.Env(),
        this,
        context,
        std::move(contextRef),
        std::move(bitmapRefs),
        std::move(rawBitmaps),
        sequenceId,
        firstTokenContextIndex,
        std::move(prompt),
        logitsLast,
        addSpecial,
        parseSpecial,
        nBatch
    );

    worker->Queue();
    return worker->GetPromise();
}

void AddonMultimodal::init(Napi::Object exports) {
    exports.Set(
        "AddonMultimodal",
        DefineClass(
            exports.Env(),
            "AddonMultimodal",
            {
                InstanceMethod("init", &AddonMultimodal::Init),
                InstanceMethod("dispose", &AddonMultimodal::Dispose),
                InstanceAccessor("disposed", &AddonMultimodal::GetDisposed, nullptr),
                InstanceMethod("supportVision", &AddonMultimodal::SupportVision),
                InstanceMethod("supportAudio", &AddonMultimodal::SupportAudio),
                InstanceMethod("supportVideo", &AddonMultimodal::SupportVideo),
                InstanceMethod("getDefaultMarker", &AddonMultimodal::GetDefaultMarker),
                InstanceMethod("createBitmapFromFile", &AddonMultimodal::CreateBitmapFromFile),
                InstanceMethod("createBitmapFromBuffer", &AddonMultimodal::CreateBitmapFromBuffer),
                InstanceMethod("evalChunks", &AddonMultimodal::EvalChunks),
            }
        )
    );
}
