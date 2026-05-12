from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        result = []
        for num in numbers:
            text = str(num)
            tokens = self.greedy_tokenize(text, vocab)
            result.append(tokens)
        return result

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        tokens = self.greedy_tokenize(text,vocab)
        return len(tokens)

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split()
        tokens = self.greedy_tokenize(text,vocab)
        return round(len(tokens)/ len(words), 4)
    
    def greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[str]:
        tokens = []
        i = 0
        while i < len(text):
            longest_match = None
            longest_end = i
                
            # try progressively shorter substrings from position i
            for j in range(len(text), i, -1):
                substring = text[i:j]
                if substring in vocab:
                    longest_match = substring
                    longest_end = j
                    break
            
            if longest_match:
                tokens.append(longest_match)
                i = longest_end
            else:
                # fallback: unknown character, skip or mark as <unk>
                tokens.append("<unk>")
                i += 1
        
        return tokens