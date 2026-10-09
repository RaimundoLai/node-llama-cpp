#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "AddonBitmap.h"
#include "AddonContext.h"
#include "AddonModel.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "napi.h"

class AddonMultimodal : public Napi::ObjectWrap<AddonMultimodal> {
    public:
        AddonModel* model = nullptr;
        mtmd_context* mtmd_ctx = nullptr;
        std::string mmprojPath;
        mtmd_context_params params;
        std::string mediaMarker;

        std::mutex disposeMutex;
        bool disposed = false;
        bool memoryDisposed = false;
        bool hasModelRef = false;
        Napi::Reference<Napi::Object> modelRef;

        AddonMultimodal(const Napi::CallbackInfo& info);
        ~AddonMultimodal();

        void disposeMemory();
        void disposeMT();

        Napi::Value Init(const Napi::CallbackInfo& info);
        Napi::Value Dispose(const Napi::CallbackInfo& info);
        Napi::Value GetDisposed(const Napi::CallbackInfo& info);
        Napi::Value SupportVision(const Napi::CallbackInfo& info);
        Napi::Value SupportAudio(const Napi::CallbackInfo& info);
        Napi::Value SupportVideo(const Napi::CallbackInfo& info);
        Napi::Value GetDefaultMarker(const Napi::CallbackInfo& info);

        Napi::Value CreateBitmapFromFile(const Napi::CallbackInfo& info);
        Napi::Value CreateBitmapFromBuffer(const Napi::CallbackInfo& info);
        Napi::Value EvalChunks(const Napi::CallbackInfo& info);

        static void init(Napi::Object exports);
};
