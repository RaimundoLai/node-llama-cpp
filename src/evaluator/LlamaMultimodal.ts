import {DisposedError, EventRelay} from "lifecycle-utils";
import fs from "fs-extra";
import {DisposeGuard} from "../utils/DisposeGuard.js";
import {MemoryMarking} from "../bindings/utils/MemoryOrchestrator.js";
import {removeNullFields} from "../utils/removeNullFields.js";
import {LlamaImage} from "./LlamaImage.js";
import type {AddonBitmap, AddonMultimodal} from "../bindings/AddonTypes.js";
import type {Token} from "../types.js";
import type {LlamaModel} from "./LlamaModel/LlamaModel.js";
import type {LlamaContextSequence} from "./LlamaContext/LlamaContext.js";
import type {SequenceEvaluateOptions} from "./LlamaContext/types.js";
import type {LlamaSampler} from "./LlamaContext/LlamaSampler.js";

export type LlamaMultimodalOptions = {
    /** Path to the multimodal projector file (`mmproj-*.gguf`) */
    mmprojPath: string,

    /** Whether to use GPU offloading for the multimodal projector. Defaults to `true`. */
    useGpu?: boolean,

    /** Whether to print timings after multimodal evaluation. Defaults to `false`. */
    printTimings?: boolean,

    /** Number of threads to use for multimodal processing. */
    nThreads?: number,

    /** Whether to perform a warmup pass after initialization. Defaults to `true`. */
    warmup?: boolean,

    /** Minimum image tokens (for dynamic resolution vision models). */
    imageMinTokens?: number,

    /** Maximum image tokens (for dynamic resolution vision models). */
    imageMaxTokens?: number,

    /** Media marker string to substitute with images. Defaults to `<__media__>`. */
    mediaMarker?: string
};

export type LlamaMultimodalMediaInputPart =
    | {type: "text", text: string}
    | {type: "image", image: string | Buffer | Uint8Array | LlamaImage}
    | {type: "audio", audio: string | Buffer | Uint8Array}
    | {type: "video", video: string | Buffer | Uint8Array, fps?: number};

export class LlamaMultimodal {
    /** @internal */ private readonly _model: LlamaModel;
    /** @internal */ private readonly _multimodal: AddonMultimodal;
    /** @internal */ private readonly _disposeGuard: DisposeGuard;
    /** @internal */ private _disposed: boolean = false;
    /** @internal */ private _disposePromise?: Promise<void>;
    /** @internal */ private _vramConsumptionMarking?: MemoryMarking;
    /** @internal */ private _ramConsumptionMarking?: MemoryMarking;
    public readonly onDispose: EventRelay<void> = new EventRelay<void>();

    private constructor(model: LlamaModel, multimodal: AddonMultimodal, estimatedMemorySize: number, useGpu: boolean) {
        this._model = model;
        this._multimodal = multimodal;
        this._disposeGuard = new DisposeGuard([model._backendModelDisposeGuard]);
        this._vramConsumptionMarking = model._llama._vramOrchestrator.markAllocation(useGpu ? estimatedMemorySize : 0);
        this._ramConsumptionMarking = model._llama._ramOrchestrator.markAllocation(useGpu ? 0 : estimatedMemorySize);
    }

    /** @internal */
    public static async _create(model: LlamaModel, options: LlamaMultimodalOptions): Promise<LlamaMultimodal> {
        const estimatedMemorySize = (await fs.stat(options.mmprojPath)).size;
        const useGpu = options.useGpu ?? true;
        const addonModel = model._model;
        const AddonMultimodalClass = model._llama._bindings.AddonMultimodal;
        const memoryReservation = useGpu
            ? model._llama._vramOrchestrator.reserveMemory(estimatedMemorySize)
            : model._llama._ramOrchestrator.reserveMemory(estimatedMemorySize);
        let addonMultimodal: AddonMultimodal | undefined;

        try {
            addonMultimodal = new AddonMultimodalClass(addonModel, removeNullFields({
                modelPath: options.mmprojPath,
                useGpu,
                printTimings: options.printTimings,
                nThreads: options.nThreads,
                warmup: options.warmup ?? true,
                imageMinTokens: options.imageMinTokens,
                imageMaxTokens: options.imageMaxTokens,
                mediaMarker: options.mediaMarker
            }));

            await addonMultimodal.init();
            return new LlamaMultimodal(model, addonMultimodal, estimatedMemorySize, useGpu);
        } catch (err) {
            addonMultimodal?.dispose();
            throw err;
        } finally {
            memoryReservation.dispose();
        }
    }

    public get model(): LlamaModel {
        return this._model;
    }

    public get supportVision(): boolean {
        if (this._disposed)
            throw new DisposedError();

        return this._multimodal.supportVision();
    }

    public get supportAudio(): boolean {
        if (this._disposed)
            throw new DisposedError();

        return this._multimodal.supportAudio();
    }

    public get supportVideo(): boolean {
        if (this._disposed)
            throw new DisposedError();

        return this._multimodal.supportVideo();
    }

    public get defaultMarker(): string {
        if (this._disposed)
            throw new DisposedError();

        return this._multimodal.getDefaultMarker();
    }

    public get disposed(): boolean {
        return this._disposed;
    }

    /** Assumed memory footprint of the multimodal projector in bytes. */
    public get memoryUsage(): {ram: number, vram: number} {
        return {
            ram: this._ramConsumptionMarking?.size ?? 0,
            vram: this._vramConsumptionMarking?.size ?? 0
        };
    }

    /** @internal */
    public get _addonMultimodal(): AddonMultimodal {
        if (this._disposed)
            throw new DisposedError();

        return this._multimodal;
    }

    /**
     * Create a LlamaImage from a file path, Buffer, Uint8Array, or base64 data URL.
     */
    public async createImage(source: string | Buffer | Uint8Array | LlamaImage): Promise<LlamaImage> {
        const preventDisposalHandle = this._createPreventDisposalHandle();
        try {
            return this._createImage(source);
        } finally {
            preventDisposalHandle.dispose();
        }
    }

    private _createImage(source: string | Buffer | Uint8Array | LlamaImage): LlamaImage {
        if (source instanceof LlamaImage)
            return source;

        if (typeof source === "string") {
            if (source.startsWith("data:")) {
                const commaIndex = source.indexOf(",");
                if (commaIndex === -1)
                    throw new Error("Invalid data URL");

                const base64Data = source.slice(commaIndex + 1);
                const buffer = Buffer.from(base64Data, "base64");
                const bitmap = this._multimodal.createBitmapFromBuffer(buffer);
                return new LlamaImage(bitmap);
            }

            const bitmap = this._multimodal.createBitmapFromFile(source);
            return new LlamaImage(bitmap);
        }

        const buffer = Buffer.isBuffer(source) ? source : Buffer.from(source);
        const bitmap = this._multimodal.createBitmapFromBuffer(buffer);
        return new LlamaImage(bitmap);
    }

    /**
     * Evaluate a prompt with images into a LlamaContextSequence.
     */
    public async evaluatePromptWithImages({
        contextSequence,
        prompt,
        images = [],
        logitsLast = true,
        nBatch,
        addSpecial,
        parseSpecial
    }: {
        contextSequence: LlamaContextSequence,
        prompt: string,
        images?: Array<string | Buffer | Uint8Array | LlamaImage>,
        logitsLast?: boolean,
        nBatch?: number,
        addSpecial?: boolean,
        parseSpecial?: boolean
    }): Promise<{
        tokensCount: number
    }> {
        const {tokensCount} = await this._evaluatePromptWithImages({
            contextSequence,
            prompt,
            images,
            logitsLast,
            nBatch,
            addSpecial,
            parseSpecial
        });
        return {tokensCount};
    }

    /** @internal */
    public async _evaluatePromptWithImagesAndSample({
        contextSequence,
        prompt,
        images = [],
        logitsLast = true,
        nBatch,
        addSpecial,
        parseSpecial,
        clearSequence = false,
        sampler,
        samplerOptions,
        signal
    }: {
        contextSequence: LlamaContextSequence,
        prompt: string,
        images?: Array<string | Buffer | Uint8Array | LlamaImage>,
        logitsLast?: boolean,
        nBatch?: number,
        addSpecial?: boolean,
        parseSpecial?: boolean,
        clearSequence?: boolean,
        sampler?: LlamaSampler,
        samplerOptions?: SequenceEvaluateOptions,
        signal?: AbortSignal
    }): Promise<{
        tokensCount: number,
        sampledToken?: Token
    }> {
        return await this._evaluatePromptWithImages({
            contextSequence,
            prompt,
            images,
            logitsLast,
            nBatch,
            addSpecial,
            parseSpecial,
            clearSequence,
            sampler,
            samplerOptions,
            signal
        });
    }

    private async _evaluatePromptWithImages({
        contextSequence,
        prompt,
        images = [],
        logitsLast = true,
        nBatch,
        addSpecial,
        parseSpecial,
        clearSequence = false,
        sampler,
        samplerOptions,
        signal
    }: {
        contextSequence: LlamaContextSequence,
        prompt: string,
        images?: Array<string | Buffer | Uint8Array | LlamaImage>,
        logitsLast?: boolean,
        nBatch?: number,
        addSpecial?: boolean,
        parseSpecial?: boolean,
        clearSequence?: boolean,
        sampler?: LlamaSampler,
        samplerOptions?: SequenceEvaluateOptions,
        signal?: AbortSignal
    }): Promise<{
        tokensCount: number,
        sampledToken?: Token
    }> {
        const preventDisposalHandle = this._createPreventDisposalHandle();

        const resolvedImages: LlamaImage[] = [];
        const imagesToDispose: LlamaImage[] = [];

        try {
            for (const img of images) {
                if (img instanceof LlamaImage) {
                    resolvedImages.push(img);
                } else {
                    const loadedImage = this._createImage(img);
                    resolvedImages.push(loadedImage);
                    imagesToDispose.push(loadedImage);
                }
            }

            return await contextSequence._evaluateMultimodalPrompt({
                multimodal: this,
                prompt,
                images: resolvedImages,
                logitsLast,
                nBatch,
                addSpecial,
                parseSpecial,
                clearSequence,
                sampler,
                samplerOptions,
                signal
            });
        } finally {
            for (const img of imagesToDispose) {
                try {
                    img.dispose();
                } catch (err) {
                    // ignore disposal error
                }
            }
            preventDisposalHandle.dispose();
        }
    }

    /**
     * Evaluate text and media parts in order. Audio accepts WAV, MP3, or FLAC;
     * video is sampled by the native llama.cpp helper at 1 frame per second by default.
     */
    public async evaluatePromptWithMedia({
        contextSequence,
        parts,
        logitsLast = true,
        nBatch,
        addSpecial,
        parseSpecial,
        clearSequence = false,
        signal
    }: {
        contextSequence: LlamaContextSequence,
        parts: readonly LlamaMultimodalMediaInputPart[],
        logitsLast?: boolean,
        nBatch?: number,
        addSpecial?: boolean,
        parseSpecial?: boolean,
        /** @internal */ clearSequence?: boolean,
        /** Stop before decoding or sampling when aborted. */ signal?: AbortSignal
    }): Promise<{
        tokensCount: number
    }> {
        const preventDisposalHandle = this._createPreventDisposalHandle();

        const promptParts: string[] = [];
        const bitmaps: AddonBitmap[] = [];
        const images: LlamaImage[] = [];
        const ownedImages: LlamaImage[] = [];
        const ownedBitmaps: AddonBitmap[] = [];

        try {
            for (const part of parts) {
                signal?.throwIfAborted();
                if (part.type === "text") {
                    promptParts.push(part.text);
                    continue;
                }

                if (part.type === "image") {
                    if (!this._multimodal.supportVision())
                        throw new Error("This multimodal projector does not support image input.");

                    const image = part.image instanceof LlamaImage
                        ? part.image
                        : this._createImage(part.image);
                    if (!(part.image instanceof LlamaImage))
                        ownedImages.push(image);
                    images.push(image);
                    bitmaps.push(image._addonBitmap);
                } else if (part.type === "audio") {
                    if (!this._multimodal.supportAudio())
                        throw new Error("This multimodal projector does not support audio input.");

                    const bitmap = typeof part.audio === "string"
                        ? this._multimodal.createBitmapFromFile(part.audio)
                        : this._multimodal.createBitmapFromBuffer(
                            Buffer.isBuffer(part.audio) ? part.audio : Buffer.from(part.audio)
                        );
                    bitmaps.push(bitmap);
                    ownedBitmaps.push(bitmap);
                } else {
                    if (!this._multimodal.supportVideo())
                        throw new Error("Video input is not available in this llama.cpp build. Build with MTMD_VIDEO and install ffmpeg and ffprobe.");

                    const fps = part.fps ?? 1;
                    if (!Number.isFinite(fps) || fps <= 0)
                        throw new RangeError("Video fps must be a positive finite number");

                    const bitmapOptions = {videoFps: fps};
                    const bitmap = typeof part.video === "string"
                        ? this._multimodal.createBitmapFromFile(part.video, bitmapOptions)
                        : this._multimodal.createBitmapFromBuffer(
                            Buffer.isBuffer(part.video) ? part.video : Buffer.from(part.video),
                            bitmapOptions
                        );
                    bitmaps.push(bitmap);
                    ownedBitmaps.push(bitmap);
                }

                promptParts.push(this.defaultMarker);
            }

            return await contextSequence._evaluateMultimodalPrompt({
                multimodal: this,
                prompt: promptParts.join(""),
                images,
                bitmaps,
                logitsLast,
                nBatch,
                addSpecial,
                parseSpecial,
                clearSequence,
                signal
            });
        } finally {
            for (const image of ownedImages) {
                try {
                    image.dispose();
                } catch (err) {
                    // ignore disposal error
                }
            }
            for (const bitmap of ownedBitmaps) {
                try {
                    bitmap.dispose();
                } catch (err) {
                    // ignore disposal error
                }
            }
            preventDisposalHandle.dispose();
        }
    }

    public dispose() {
        void this._disposeWithGuard();
    }

    public [Symbol.dispose]() {
        this.dispose();
    }

    public [Symbol.asyncDispose]() {
        return this._disposeWithGuard();
    }

    /** @internal */
    public _createPreventDisposalHandle() {
        if (this._disposed)
            throw new DisposedError();

        return this._disposeGuard.createPreventDisposalHandle();
    }

    /** @internal */
    private _disposeWithGuard() {
        if (this._disposePromise != null)
            return this._disposePromise;

        this._disposed = true;
        this._disposePromise = (async () => {
            await this._disposeGuard.acquireDisposeLock();
            this._multimodal.dispose();
            this._vramConsumptionMarking?.dispose();
            this._vramConsumptionMarking = undefined;
            this._ramConsumptionMarking?.dispose();
            this._ramConsumptionMarking = undefined;
            this.onDispose.dispatchEvent();
        })();

        return this._disposePromise;
    }
}
