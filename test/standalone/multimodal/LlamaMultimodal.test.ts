import path from "path";
import os from "os";
import {fileURLToPath} from "url";
import fs from "fs-extra";
import {describe, expect, test} from "vitest";
import {DisposedError} from "lifecycle-utils";
import {LlamaImage, LlamaMultimodal} from "../../../src/index.js";
import {getTestLlama} from "../../utils/getTestLlama.js";
import type {LlamaModel} from "../../../src/evaluator/LlamaModel/LlamaModel.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const testImageFilePath = path.join(__dirname, "..", "..", "..", "llama", "llama.cpp", "tools", "mtmd", "test-1.jpeg");

describe("multimodal", () => {
    test("bindings export AddonBitmap and AddonMultimodal", async () => {
        const llama = await getTestLlama();
        expect(llama._bindings.AddonBitmap).toBeDefined();
        expect(llama._bindings.AddonMultimodal).toBeDefined();
    });

    test("LlamaImage loads and inspects a JPEG image from buffer", async () => {
        const llama = await getTestLlama();
        const imageBuffer = await fs.readFile(testImageFilePath);

        const bitmap = llama._bindings.AddonBitmap.initFromBuffer(imageBuffer);
        const image = new LlamaImage(bitmap);

        expect(image.disposed).toBe(false);
        expect(image.width).toBeGreaterThan(0);
        expect(image.height).toBeGreaterThan(0);
        expect(image.bytes).toBeGreaterThan(0);
        // The helper assigns a content hash used by the vision cache.
        expect(image.id).toEqual(expect.any(String));

        image.id = "test-image-id";
        expect(image.id).toBe("test-image-id");

        const clone = image.clone();
        expect(clone.id).toBe(image.id);
        expect(clone.width).toBe(image.width);
        image.id = null;
        expect(image.id).toBeNull();
        expect(clone.id).toBe("test-image-id");
        await clone[Symbol.asyncDispose]();

        image.dispose();
        expect(image.disposed).toBe(true);

        // Idempotent dispose
        image.dispose();
        expect(image.disposed).toBe(true);

        // Accessing properties after dispose throws DisposedError
        expect(() => image.width).toThrow(DisposedError);
        expect(() => image.height).toThrow(DisposedError);
        expect(() => image.bytes).toThrow(DisposedError);
        expect(() => image.id).toThrow(DisposedError);
    });

    test("LlamaImage loads from file path directly", async () => {
        const llama = await getTestLlama();
        const bitmap = llama._bindings.AddonBitmap.initFromFile(testImageFilePath);
        const image = new LlamaImage(bitmap);

        expect(image.disposed).toBe(false);
        expect(image.width).toBeGreaterThan(0);
        expect(image.height).toBeGreaterThan(0);

        image.dispose();
        expect(image.disposed).toBe(true);
    });

    test("LlamaImage Symbol.dispose cleans up resource with using syntax", async () => {
        const llama = await getTestLlama();
        const imageBuffer = await fs.readFile(testImageFilePath);

        let imageRef: LlamaImage;
        {
            const bitmap = llama._bindings.AddonBitmap.initFromBuffer(imageBuffer);
            using image = new LlamaImage(bitmap);
            imageRef = image;
            expect(image.disposed).toBe(false);
        }
        expect(imageRef!.disposed).toBe(true);
    });

    test("AddonBitmap throws error on invalid buffer", async () => {
        const llama = await getTestLlama();
        const invalidBuffer = Buffer.from([0, 1, 2, 3, 4, 5]);

        expect(() => {
            llama._bindings.AddonBitmap.initFromBuffer(invalidBuffer);
        }).toThrow();
    });

    test("AddonBitmap throws error on non-existent file", async () => {
        const llama = await getTestLlama();
        expect(() => {
            llama._bindings.AddonBitmap.initFromFile("/nonexistent/image.png");
        }).toThrow();
    });

    test("LlamaMultimodal propagates projector initialization errors", async () => {
        const error = new Error("Failed to load multimodal projector");
        const testDirectory = await fs.mkdtemp(path.join(os.tmpdir(), "node-llama-cpp-multimodal-"));
        const mmprojPath = path.join(testDirectory, "mmproj.gguf");

        class AddonMultimodalStub {
            public async init() {
                throw error;
            }

            public dispose() {}
        }
        const model = {
            _model: {},
            _llama: {
                _bindings: {
                    AddonMultimodal: AddonMultimodalStub
                },
                _vramOrchestrator: {
                    reserveMemory: () => ({dispose() {}}),
                    markAllocation: () => ({dispose() {}})
                }
            }
        } as unknown as LlamaModel;

        try {
            await fs.writeFile(mmprojPath, "test projector placeholder");
            await expect(
                LlamaMultimodal._create(model, {mmprojPath})
            ).rejects.toBe(error);
        } finally {
            await fs.remove(testDirectory);
        }
    });
});
