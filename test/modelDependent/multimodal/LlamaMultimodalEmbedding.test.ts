import path from "path";
import os from "os";
import {fileURLToPath} from "url";
import {execFileSync} from "child_process";
import fs from "fs-extra";
import {afterAll, beforeAll, describe, expect, test} from "vitest";
import {getLlama, Llama, LlamaModel, LlamaEmbeddingContext} from "../../../src/index.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const modelPath = path.join(root, "embeddinggemma-2-Q8_0.gguf");
const mmprojPath = path.join(root, "mmproj-Q8_0.gguf");
const fixturePath = path.join(root, "assets");

describe.skipIf(!fs.existsSync(modelPath) || !fs.existsSync(mmprojPath))("EmbeddingGemma 2 media", () => {
    let llama: Llama;
    let model: LlamaModel;
    let context: LlamaEmbeddingContext;
    let temporary: string;

    beforeAll(async () => {
        temporary = await fs.mkdtemp(path.join(os.tmpdir(), "nlc-media-embedding-"));
        execFileSync("ffmpeg", ["-nostdin", "-v", "error", "-y", "-i", path.join(fixturePath, "test-03.wav"),
            "-t", "2", "-ac", "1", "-ar", "16000", path.join(temporary, "audio.wav")], {timeout: 30000});
        execFileSync("ffmpeg", ["-nostdin", "-v", "error", "-y", "-i", path.join(fixturePath, "output-1791335893029.mp4"),
            "-t", "2", "-an", "-vf", "fps=1,scale=640:-2", "-c:v", "libx264", "-preset", "ultrafast",
            path.join(temporary, "video.mp4")], {timeout: 30000});
        execFileSync("ffmpeg", ["-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=224x224",
            "-frames:v", "1", path.join(temporary, "red.png")], {timeout: 30000});
        llama = await getLlama("lastBuild", {logLevel: "error"});
        model = await llama.loadModel({modelPath, gpuLayers: 0});
        await model.loadMultimodal({mmprojPath, useGpu: false, warmup: false, imageMaxTokens: 280});
        context = await model.createEmbeddingContext({contextSize: 8192, batchSize: 8192});
    }, 60000);

    afterAll(async () => {
        await context?.dispose();
        await model?.dispose();
        await llama?.dispose();
        if (temporary != null)
            await fs.remove(temporary);
    });

    test("evaluates full text, images, audio, video and mixed input", async () => {
        const first = await context.getEmbeddingForMultimodal([{type: "text", text: "task: search result | query: a red car"}]);
        const second = await context.getEmbeddingForMultimodal([{type: "text", text: "task: search result | query: ocean waves"}]);
        expect(first.vector).toHaveLength(768);
        expect(first.vector).not.toEqual(second.vector);
        for (const parts of [
            [{type: "image" as const, image: path.join(fixturePath, "logo.png")}],
            [{type: "audio" as const, audio: path.join(temporary, "audio.wav")}],
            [{type: "video" as const, video: path.join(temporary, "video.mp4"), fps: 1}],
            [{type: "text" as const, text: "A subtitle tutorial: "},
                {type: "image" as const, image: path.join(fixturePath, "logo.png")}]
        ]) {
            const embedding = await context.getEmbeddingForMultimodal(parts);
            expect(embedding.vector).toHaveLength(768);
            expect(embedding.vector.every(Number.isFinite)).toBe(true);
            expect(Math.hypot(...embedding.vector)).toBeGreaterThan(0);
        }
    }, 60000);

    test("rejects cancelled requests without losing subsequent evaluations", async () => {
        const controller = new AbortController();
        controller.abort();
        await expect(context.getEmbeddingForMultimodal([{type: "text", text: "cancelled"}], {signal: controller.signal}))
            .rejects.toHaveProperty("name", "AbortError");
        expect((await context.getEmbeddingForMultimodal([{type: "text", text: "next request"}])).vector).toHaveLength(768);
    });

    test("pools the media together with trailing text instead of only the last chunk", async () => {
        const suffix = {type: "text" as const, text: "This is the same trailing text."};
        const first = await context.getEmbeddingForMultimodal([
            {type: "image", image: path.join(temporary, "red.png")}, suffix
        ]);
        const second = await context.getEmbeddingForMultimodal([
            {type: "image", image: path.join(fixturePath, "logo.png")}, suffix
        ]);
        expect(first.vector).not.toEqual(second.vector);
        expect(first.calculateCosineSimilarity(second)).toBeLessThan(0.999);
    }, 30000);

    test("retrieves the subtitle tutorial image with a related text query", async () => {
        const image = await context.getEmbeddingForMultimodal([
            {type: "image", image: path.join(fixturePath, "logo.png")}
        ]);
        const related = await context.getEmbeddingForMultimodal([
            {type: "text", text: "task: search result | query: subtitle synchronization tutorial"}
        ]);
        const unrelated = await context.getEmbeddingForMultimodal([
            {type: "text", text: "task: search result | query: a dog running in the snow"}
        ]);
        expect(image.calculateCosineSimilarity(related)).toBeGreaterThan(image.calculateCosineSimilarity(unrelated));
    }, 30000);

    test("requires a complete embedding batch and recovers after an undersized batch", async () => {
        const smallBatch = await model.createEmbeddingContext({contextSize: 8192, batchSize: 128});
        try {
            await expect(smallBatch.getEmbeddingForMultimodal([
                {type: "image", image: path.join(fixturePath, "logo.png")}
            ]))
                .rejects.toThrow("batchSize to fit the complete input");
            expect((await smallBatch.getEmbeddingForMultimodal([{type: "text", text: "short input"}])).vector).toHaveLength(768);
        } finally {
            await smallBatch.dispose();
        }
    }, 30000);

    test("rejects context overflow before native decoding and recovers", async () => {
        const small = await model.createEmbeddingContext({contextSize: 128, batchSize: 128});
        try {
            await expect(small.getEmbeddingForMultimodal([{type: "text", text: "long input ".repeat(200)}]))
                .rejects.toThrow("exceeds the sequence context size");
            expect((await small.getEmbeddingForMultimodal([{type: "text", text: "short input"}])).vector).toHaveLength(768);
        } finally {
            await small.dispose();
        }
    });
});
