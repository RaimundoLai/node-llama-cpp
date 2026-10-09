#include "AddonBitmap.h"
#include "addonGlobals.h"
#include "mtmd-helper.h"
#include <cstring>
#include <filesystem>
#include <fstream>

namespace {
bool isAudioInput(const unsigned char* data, size_t length) {
    if (length < 12) {
        return false;
    }

    const bool isWav = std::memcmp(data, "RIFF", 4) == 0 && std::memcmp(data + 8, "WAVE", 4) == 0;
    const bool isMp3 = std::memcmp(data, "ID3", 3) == 0 ||
        (data[0] == 0xFF && (data[1] & 0xE0) == 0xE0);
    const bool isFlac = std::memcmp(data, "fLaC", 4) == 0;
    return isWav || isMp3 || isFlac;
}

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
}

Napi::FunctionReference AddonBitmap::constructor;

AddonBitmap::AddonBitmap(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<AddonBitmap>(info) {
    if (info.Length() > 0 && info[0].IsExternal()) {
        bitmap = info[0].As<Napi::External<mtmd_bitmap>>().Data();
    }
    if (info.Length() > 1 && info[1].IsExternal()) {
        videoContext = info[1].As<Napi::External<mtmd_helper_video>>().Data();
    }

    if (bitmap != nullptr) {
        externalMemorySize = mtmd_bitmap_get_n_bytes(bitmap);
        if (externalMemorySize > 0) {
            adjustNapiExternalMemoryAdd(info.Env(), externalMemorySize);
        }
    }
}

AddonBitmap::~AddonBitmap() {
    disposeMT();
}

void AddonBitmap::disposeMemory() {
    mtmd_bitmap* currentBitmap = nullptr;
    mtmd_helper_video* currentVideoContext = nullptr;
    {
        std::lock_guard<std::mutex> lock(disposeMutex);
        if (memoryDisposed) {
            return;
        }
        memoryDisposed = true;
        currentBitmap = bitmap;
        currentVideoContext = videoContext;
        bitmap = nullptr;
        videoContext = nullptr;
    }

    if (currentVideoContext != nullptr) {
        mtmd_helper_video_free(currentVideoContext);
    }
    if (currentBitmap != nullptr) {
        mtmd_bitmap_free(currentBitmap);
    }
}

void AddonBitmap::disposeMT() {
    uint64_t currentSize = 0;
    {
        std::lock_guard<std::mutex> lock(disposeMutex);
        if (disposed) {
            return;
        }
        disposed = true;
        currentSize = externalMemorySize;
        externalMemorySize = 0;
    }

    disposeMemory();

    if (currentSize > 0) {
        adjustNapiExternalMemorySubtract(Env(), currentSize);
    }
}

Napi::Value AddonBitmap::Dispose(const Napi::CallbackInfo& info) {
    disposeMT();
    return info.Env().Undefined();
}

Napi::Value AddonBitmap::GetDisposed(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), disposed);
}

Napi::Value AddonBitmap::GetWidth(const Napi::CallbackInfo& info) {
    if (disposed || bitmap == nullptr) {
        Napi::Error::New(info.Env(), "Bitmap is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
    return Napi::Number::New(info.Env(), mtmd_bitmap_get_nx(bitmap));
}

Napi::Value AddonBitmap::GetHeight(const Napi::CallbackInfo& info) {
    if (disposed || bitmap == nullptr) {
        Napi::Error::New(info.Env(), "Bitmap is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
    return Napi::Number::New(info.Env(), mtmd_bitmap_get_ny(bitmap));
}

Napi::Value AddonBitmap::GetBytes(const Napi::CallbackInfo& info) {
    if (disposed || bitmap == nullptr) {
        Napi::Error::New(info.Env(), "Bitmap is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
    return Napi::Number::New(info.Env(), mtmd_bitmap_get_n_bytes(bitmap));
}

Napi::Value AddonBitmap::GetId(const Napi::CallbackInfo& info) {
    if (disposed || bitmap == nullptr) {
        Napi::Error::New(info.Env(), "Bitmap is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
    const char* id = mtmd_bitmap_get_id(bitmap);
    if (id == nullptr || strlen(id) == 0) {
        return info.Env().Null();
    }
    return Napi::String::New(info.Env(), id);
}

void AddonBitmap::SetId(const Napi::CallbackInfo& info, const Napi::Value& value) {
    if (disposed || bitmap == nullptr) {
        Napi::Error::New(info.Env(), "Bitmap is disposed").ThrowAsJavaScriptException();
        return;
    }
    if (value.IsNull() || value.IsUndefined()) {
        mtmd_bitmap_set_id(bitmap, nullptr);
    } else {
        std::string id = value.As<Napi::String>().Utf8Value();
        mtmd_bitmap_set_id(bitmap, id.c_str());
    }
}

Napi::Value AddonBitmap::SetMergeable(const Napi::CallbackInfo& info) {
    if (disposed || bitmap == nullptr) {
        Napi::Error::New(info.Env(), "Bitmap is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
    if (info.Length() < 1 || !info[0].IsBoolean()) {
        Napi::TypeError::New(info.Env(), "Expected a boolean mergeable value").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    mtmd_bitmap_set_mergeable(bitmap, info[0].As<Napi::Boolean>().Value());
    return info.Env().Undefined();
}

Napi::Value AddonBitmap::Clone(const Napi::CallbackInfo& info) {
    if (disposed || bitmap == nullptr) {
        Napi::Error::New(info.Env(), "Bitmap is disposed").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    mtmd_bitmap* clonedBitmap = mtmd_bitmap_is_audio(bitmap)
        ? mtmd_bitmap_init_from_audio(
            mtmd_bitmap_get_nx(bitmap),
            reinterpret_cast<const float*>(mtmd_bitmap_get_data(bitmap))
        )
        : mtmd_bitmap_init(
            mtmd_bitmap_get_nx(bitmap),
            mtmd_bitmap_get_ny(bitmap),
            mtmd_bitmap_get_data(bitmap)
        );
    if (clonedBitmap == nullptr) {
        Napi::Error::New(info.Env(), "Failed to clone bitmap").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    const char* id = mtmd_bitmap_get_id(bitmap);
    if (id != nullptr && strlen(id) > 0)
        mtmd_bitmap_set_id(clonedBitmap, id);

    return AddonBitmap::NewInstance(info.Env(), clonedBitmap);
}

void AddonBitmap::init(Napi::Object exports) {
    Napi::Function func = DefineClass(
        exports.Env(),
        "AddonBitmap",
        {
            InstanceMethod("dispose", &AddonBitmap::Dispose),
            InstanceAccessor("disposed", &AddonBitmap::GetDisposed, nullptr),
            InstanceAccessor("width", &AddonBitmap::GetWidth, nullptr),
            InstanceAccessor("height", &AddonBitmap::GetHeight, nullptr),
            InstanceAccessor("bytes", &AddonBitmap::GetBytes, nullptr),
            InstanceAccessor("id", &AddonBitmap::GetId, &AddonBitmap::SetId),
            InstanceMethod("setMergeable", &AddonBitmap::SetMergeable),
            InstanceMethod("clone", &AddonBitmap::Clone),
            StaticMethod("initFromBuffer", &AddonBitmap::InitFromBuffer),
            StaticMethod("initFromFile", &AddonBitmap::InitFromFile),
        }
    );
    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();
    exports.Set("AddonBitmap", func);
}

Napi::Object AddonBitmap::NewInstance(Napi::Env env, mtmd_bitmap* bitmap, mtmd_helper_video* videoContext) {
    if (videoContext == nullptr) {
        return constructor.New({Napi::External<mtmd_bitmap>::New(env, bitmap)});
    }
    return constructor.New({
        Napi::External<mtmd_bitmap>::New(env, bitmap),
        Napi::External<mtmd_helper_video>::New(env, videoContext)
    });
}

Napi::Value AddonBitmap::InitFromBuffer(const Napi::CallbackInfo& info) {
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
    if (isAudioInput(data, length)) {
        Napi::Error::New(info.Env(), "Audio input requires an initialized multimodal context").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    auto bitmapWrapper = mtmd_helper_bitmap_init_from_buf(
        nullptr,
        data,
        length,
        false,
        mtmd_helper_init_opt_default()
    );
    if (bitmapWrapper.bitmap == nullptr) {
        Napi::Error::New(info.Env(), "Failed to create bitmap from buffer").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    return AddonBitmap::NewInstance(info.Env(), bitmapWrapper.bitmap, bitmapWrapper.video_ctx);
}

Napi::Value AddonBitmap::InitFromFile(const Napi::CallbackInfo& info) {
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(info.Env(), "Expected string file path as first argument").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    std::string filePath = info[0].As<Napi::String>().Utf8Value();
    unsigned char header[12];
    std::ifstream file(std::filesystem::u8path(filePath), std::ios::binary);
    if (file.read(reinterpret_cast<char*>(header), sizeof(header)) && isAudioInput(header, sizeof(header))) {
        Napi::Error::New(info.Env(), "Audio input requires an initialized multimodal context").ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }
    file.close();

    auto bitmapWrapper = mtmd_helper_bitmap_init_from_file(
        nullptr,
        filePath.c_str(),
        false,
        mtmd_helper_init_opt_default()
    );
    if (bitmapWrapper.bitmap == nullptr) {
        Napi::Error::New(info.Env(), "Failed to create bitmap from file: " + filePath).ThrowAsJavaScriptException();
        return info.Env().Undefined();
    }

    return AddonBitmap::NewInstance(info.Env(), bitmapWrapper.bitmap, bitmapWrapper.video_ctx);
}
