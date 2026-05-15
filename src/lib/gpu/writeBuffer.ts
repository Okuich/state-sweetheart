/**
 * Type-safe wrapper around `device.queue.writeBuffer`.
 *
 * The WebGPU type for the third argument is `BufferSource`
 * (= `ArrayBufferView | ArrayBuffer`). TypeScript's `lib.dom`
 * versions of `Uint32Array`, `Float32Array`, etc. don't always
 * structurally match `BufferSource` across TS versions / lib
 * targets, which forces `as unknown as BufferSource` casts at
 * every call site.
 *
 * `writeTypedBuffer` accepts any typed array or ArrayBuffer and
 * does the cast in exactly one place, so call sites stay clean
 * and a future TS upgrade only needs to be fixed here.
 *
 * Usage:
 *   writeTypedBuffer(device, buf, data);
 *   writeTypedBuffer(device, buf, data, 16);   // bufferOffset
 *
 * For sub-range uploads (dataOffset / size), pass a subarray:
 *   writeTypedBuffer(device, buf, data.subarray(start, end));
 */
export type GpuUploadable = ArrayBufferView | ArrayBuffer;

export function writeTypedBuffer(
  device: GPUDevice,
  buffer: GPUBuffer,
  data: GpuUploadable,
  bufferOffset = 0,
): void {
  device.queue.writeBuffer(buffer, bufferOffset, data as unknown as BufferSource);
}
