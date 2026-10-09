#pragma once

#include <mutex>
#include "mtmd.h"
#include "napi.h"

struct mtmd_helper_video;

class AddonBitmap : public Napi::ObjectWrap<AddonBitmap> {
    public:
        mtmd_bitmap* bitmap = nullptr;
        mtmd_helper_video* videoContext = nullptr;
        std::mutex disposeMutex;
        bool disposed = false;
        bool memoryDisposed = false;
        uint64_t externalMemorySize = 0;

        AddonBitmap(const Napi::CallbackInfo& info);
        ~AddonBitmap();

        void disposeMemory();
        void disposeMT();

        Napi::Value Dispose(const Napi::CallbackInfo& info);
        Napi::Value GetDisposed(const Napi::CallbackInfo& info);
        Napi::Value GetWidth(const Napi::CallbackInfo& info);
        Napi::Value GetHeight(const Napi::CallbackInfo& info);
        Napi::Value GetBytes(const Napi::CallbackInfo& info);
        Napi::Value GetId(const Napi::CallbackInfo& info);
        void SetId(const Napi::CallbackInfo& info, const Napi::Value& value);
        Napi::Value SetMergeable(const Napi::CallbackInfo& info);
        Napi::Value Clone(const Napi::CallbackInfo& info);

        static void init(Napi::Object exports);
        static Napi::Object NewInstance(Napi::Env env, mtmd_bitmap* bitmap, mtmd_helper_video* videoContext = nullptr);
        static Napi::Value InitFromBuffer(const Napi::CallbackInfo& info);
        static Napi::Value InitFromFile(const Napi::CallbackInfo& info);

    private:
        static Napi::FunctionReference constructor;
};
