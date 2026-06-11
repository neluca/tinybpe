# TinyBPE Algorithm

## Training (BPE Learning)

The trainer learns merge pairs from byte-level token sequences using a frequency-based greedy algorithm:

1. **Initialization**: The training text is split into pieces (chunks). Each byte becomes a base token ID (0–255).

2. **Training step** (repeated for each merge):
   - Scan all adjacent token pairs across all pieces
   - Count frequencies using an AVL tree (`O(n log n)`)
   - Find the pair with the highest count (linear scan over unique pairs)
   - Replace all occurrences of the winning pair with a new token ID
   - The new token ID becomes available for future merges

3. **Termination**: Training stops when no more pairs exist or the desired vocabulary size is reached.

### Data Structure

The AVL tree stores `(left, right, frequency)` nodes. The balance factor is packed into the low 2 bits of each node's parent pointer, avoiding extra memory per node.

### Time Complexity

- Per step: `O(N log U)` where `N` = total adjacent pairs, `U` = unique pairs
- Total for `K` merges on text of length `T`: `O(K · T log T)`

## Encoding (Text → Token IDs)

Greedy lowest-rank-first merging:

1. Convert each input byte to a base token ID (0–255)
2. Repeat until no more merges:
   - Look up the merge rank of every adjacent pair in the merges AVL tree (`O(log M)` per lookup)
   - Find the pair with the smallest (lowest) rank
   - Replace all occurrences of that pair with its merged token ID
3. Return the final token ID sequence

### Determinism

The greedy approach ensures deterministic encoding: the same merge list always produces the same token IDs for the same input.

## Decoding (Token IDs → Text)

Direct `O(1)` per token lookup: the flat `bpe_vocab` array maps each token ID to its byte sequence. All token bytes are stored in a single contiguous allocation for cache efficiency.

## Streaming Decode

Processes one token ID at a time through a 4-byte cache that reassembles UTF-8 characters spanning token boundaries. Complete characters are returned immediately; incomplete bytes are held in the cache for the next call.

## References

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2016)
- [GPT-2 tokenizer](https://github.com/openai/gpt-2) (OpenAI)
