def generate_tikz_alignment(french_sentence, english_sentence, alignment_str):
    
    fr_tokens = french_sentence.strip().split()
    en_tokens = english_sentence.strip().split()
    links = [tuple(map(int, pair.split("-"))) for pair in alignment_str.strip().split()]

    latex = []
    latex.append("\\begin{tikzpicture}[baseline]")
    latex.append("\\tikzset{every node/.style={font=\\small}}")

    for i, w in enumerate(fr_tokens):
        latex.append(f"\\node (f{i}) at ({i}, 1) {{{w}}};")
    for j, w in enumerate(en_tokens):
        latex.append(f"\\node (e{j}) at ({j}, 0) {{{w}}};")

    # Alignment lines
    for (i, j) in links:
        if j<len(fr_tokens) and i<len(en_tokens):
            latex.append(f"\\draw[->, thick, gray!70] (f{j}.south) -- (e{i}.north);")

    latex.append("\\end{tikzpicture}")

    return "\n".join(latex)


if __name__ == '__main__':
    french = "je vais voyager"
    english = "i 'm going to travel"
    align_str = "0-0 0-0 1-0 1-1 2-1 2-1 3-1 3-2 4-2 4-3 5-2 5-4"

    tikz_code = generate_tikz_alignment(french, english, align_str)
    print(tikz_code)
