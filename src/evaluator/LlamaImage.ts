import {DisposedError, EventRelay} from "lifecycle-utils";
import {DisposeGuard} from "../utils/DisposeGuard.js";
import type {AddonBitmap} from "../bindings/AddonTypes.js";

export class LlamaImage {
    /** @internal */ private readonly _bitmap: AddonBitmap;
    /** @internal */ private readonly _disposeGuard: DisposeGuard = new DisposeGuard();
    /** @internal */ private _disposed: boolean = false;
    /** @internal */ private _disposePromise?: Promise<void>;
    public readonly onDispose: EventRelay<void> = new EventRelay<void>();

    public constructor(bitmapOrOptions: AddonBitmap | {_bitmap: AddonBitmap}) {
        if ("_bitmap" in bitmapOrOptions)
            this._bitmap = bitmapOrOptions._bitmap;
        else
            this._bitmap = bitmapOrOptions;
    }

    public get width(): number {
        if (this._disposed)
            throw new DisposedError();

        return this._bitmap.width;
    }

    public get height(): number {
        if (this._disposed)
            throw new DisposedError();

        return this._bitmap.height;
    }

    public get bytes(): number {
        if (this._disposed)
            throw new DisposedError();

        return this._bitmap.bytes;
    }

    public get id(): string | null {
        if (this._disposed)
            throw new DisposedError();

        return this._bitmap.id;
    }

    public set id(value: string | null) {
        if (this._disposed)
            throw new DisposedError();

        this._bitmap.id = value;
    }

    public get disposed(): boolean {
        return this._disposed;
    }

    /** Create an independent copy of this image. */
    public clone(): LlamaImage {
        if (this._disposed)
            throw new DisposedError();

        return new LlamaImage(this._bitmap.clone());
    }

    /** @internal */
    public get _addonBitmap(): AddonBitmap {
        if (this._disposed)
            throw new DisposedError();

        return this._bitmap;
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
            this._bitmap.dispose();
            this.onDispose.dispatchEvent();
        })();

        return this._disposePromise;
    }
}
