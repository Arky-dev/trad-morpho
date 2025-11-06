import torch
import torch.nn.functional as F
from model import MorphAwareTranslator

def translate_sentence(model, sentence_tokens, vocab_fr, vocab_en, device):
    # Map words to IDs
    ids = [vocab_fr.get(tok, vocab_fr["<unk>"]) for tok in sentence_tokens]
    x_surface = torch.tensor([ids], device=device)

    # Dummy morphological features (if not available at test time)
    x_morph = {f: torch.zeros_like(x_surface) for f in model.morph_embs.keys()}

    model.eval()
    with torch.no_grad():
        surf = model.surface_emb(x_surface)
        morph_vecs = [model.morph_embs[f](x_morph[f]) for f in x_morph]
        morph_cat = torch.cat(morph_vecs, dim=-1)
        enc_in = torch.cat([surf, morph_cat], dim=-1)
        enc_out, _ = model.encoder(enc_in)
        enc_out = model.enc_proj(enc_out)
        h = enc_out.mean(dim=1).unsqueeze(0)

        sos = torch.tensor([[1]], device=device)
        dec_input = model.en_emb(sos)
        outputs = []

        for _ in range(40):
            h_rep = h[-1].unsqueeze(1).repeat(1, enc_out.size(1), 1)
            energy = torch.tanh(model.attn(torch.cat([h_rep, enc_out], dim=-1)))
            scores = model.v(energy).squeeze(-1)
            a_t = F.softmax(scores, dim=-1)
            context = torch.bmm(a_t.unsqueeze(1), enc_out)
            dec_input_full = torch.cat([dec_input, context], dim=-1)
            out, h = model.decoder(dec_input_full, h)
            logits = model.fc_out(out[:, -1])
            next_tok = logits.argmax(-1)
            outputs.append(next_tok.item())
            if next_tok.item() == 2:  # <eos>
                break
            dec_input = model.en_emb(next_tok.unsqueeze(0))

        inv_vocab = {v: k for k, v in vocab_en.items()}
        return " ".join(inv_vocab.get(i, "<unk>") for i in outputs)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load("checkpoints/model.pt", map_location=device)
    vocabs = checkpoint["vocabs"]
    config = checkpoint["config"]

    vocab_sizes = {
        "surface": len(vocabs["surface"]),
        "en": len(vocabs["en"]),
        "morph": {f: len(v) for f, v in vocabs["morph"].items()}
    }

    # Build model with sizes, then load weights
    model = MorphAwareTranslator(vocab_sizes, **config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    sentence = input("Enter a French sentence: ").strip().split()
    translation = translate_sentence(model, sentence, vocabs["surface"], vocabs["en"], device)
    print("\nEnglish translation:\n", translation)

if __name__ == "__main__":
    main()
