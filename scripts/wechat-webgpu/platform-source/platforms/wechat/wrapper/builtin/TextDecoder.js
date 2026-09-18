// UTF-8 subset of TextDecoder for the shader WASM glue. Preserve native support.
// Supports BufferSource offsets, BOM, fatal decoding and split streaming input.
class UTF8TextDecoder {
  constructor(label = 'utf-8', options = {}) {
    if (!['utf-8', 'utf8', 'unicode-1-1-utf-8'].includes(String(label).trim().toLowerCase())) {
      throw new RangeError('Only UTF-8 TextDecoder is supported by this adapter');
    }
    this.encoding = 'utf-8';
    this.fatal = !!options.fatal;
    this.ignoreBOM = !!options.ignoreBOM;
    this._pending = [];
    this._bomSeen = false;
    this._streaming = false;
  }

  decode(input, options = {}) {
    let bytes;
    if (input === undefined) bytes = new Uint8Array(0);
    else if (ArrayBuffer.isView(input)) bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    else if (Object.prototype.toString.call(input) === '[object ArrayBuffer]'
      || Object.prototype.toString.call(input) === '[object SharedArrayBuffer]') bytes = new Uint8Array(input);
    else throw new TypeError('TextDecoder input must be a BufferSource');
    if (!this._streaming) {
      this._pending = [];
      this._bomSeen = false;
    }
    const data = this._pending.concat(Array.from(bytes));
    this._pending = [];
    this._streaming = !!options.stream;
    let output = '';
    const emit = cp => {
      if (!this._bomSeen) {
        this._bomSeen = true;
        if (cp === 0xfeff && !this.ignoreBOM) return;
      }
      output += String.fromCodePoint(cp);
    };
    const invalid = () => {
      if (this.fatal) {
        this._pending = [];
        this._streaming = false;
        throw new TypeError('Invalid UTF-8 data');
      }
      emit(0xfffd);
    };
    for (let i = 0; i < data.length;) {
      const start = i;
      const first = data[i++];
      if (first < 0x80) { emit(first); continue; }
      let remaining, cp, lower = 0x80, upper = 0xbf;
      if (first >= 0xc2 && first <= 0xdf) { remaining = 1; cp = first & 0x1f; }
      else if (first >= 0xe0 && first <= 0xef) {
        remaining = 2; cp = first & 0x0f;
        if (first === 0xe0) lower = 0xa0;
        if (first === 0xed) upper = 0x9f;
      } else if (first >= 0xf0 && first <= 0xf4) {
        remaining = 3; cp = first & 0x07;
        if (first === 0xf0) lower = 0x90;
        if (first === 0xf4) upper = 0x8f;
      } else { invalid(); continue; }
      let complete = true;
      for (let j = 0; j < remaining; j++) {
        if (i === data.length) {
          if (this._streaming) this._pending = data.slice(start);
          else invalid();
          complete = false;
          break;
        }
        const next = data[i];
        if (next < lower || next > upper) {
          invalid(); // Reprocess the offending byte as a new sequence.
          complete = false;
          break;
        }
        cp = (cp << 6) | (next & 0x3f);
        i++;
        lower = 0x80; upper = 0xbf;
      }
      if (complete) emit(cp);
    }
    return output;
  }
}

const decoderGlobals = [globalThis];
if (typeof GameGlobal !== 'undefined') decoderGlobals.push(GameGlobal);
const decoder = decoderGlobals.map(host => host.TextDecoder).find(value => typeof value === 'function') || UTF8TextDecoder;
decoderGlobals.forEach(host => {
  if (typeof host.TextDecoder !== 'function') host.TextDecoder = decoder;
});
