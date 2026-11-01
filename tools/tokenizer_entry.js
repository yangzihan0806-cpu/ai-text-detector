import { Tiktoken } from "js-tiktoken/lite";
import gpt2 from "js-tiktoken/ranks/gpt2";
window.gpt2Tokenizer = new Tiktoken(gpt2);
