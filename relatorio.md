# Trabalho de Implementação 2

Marcos Samuel Winkel Landi - 00304688

## 1 - a)

Comparando os tamanhos de vizinhança, é possível ver que com uma vizinhança menor o detalhe é maior mas o ruído também, ao passo que com uma vizinhança maior, as texturas são mais suaves porém o detalhe é diminuido, como pode-se observar nas pontas dos chapéus de aniversário.

<img src="dispSSDfull.png" alt="dispSSDfull" style="zoom: 48%;" />

## 1 - b)

Foi usada a distância soma dos mínimos quadrados no espaço de cores L\*a\*b\*, que é um espaço de cores mais perceptualmente uniformes que RGB ou BGR.

<img src="C:\Users\Marcos\Desktop\facul\visao\T2\LAB_Comparison_Full.png" alt="LAB_Comparison_Full" style="zoom:51%;" />

As diferenças não são facilmente perceptíveis ao compararmos os resultados lado a lado, por isso gifs comparativos podem ser encontrados em [github.com/mswlandi/stereoAssignment/blob/master/gifSources/gifs.md](github.com/mswlandi/stereoAssignment/blob/master/gifSources/gifs.md).

## 2

Variando N e mantendo M=3, com a coluna da esquerda sendo as referências que não usam processos de agregação:

- Média:

<img src="C:\Users\Marcos\Desktop\facul\visao\stereoAssignment\dispMeanAgg_Comparison_Full.png" alt="dispMeanAgg_Comparison_Full" style="zoom:50%;" />

- Mediana:

<img src="C:\Users\Marcos\Desktop\facul\visao\stereoAssignment\dispMedianAgg_Comparison_Full.png" alt="dispMedianAgg_Comparison_Full" style="zoom:50%;" />

A maior melhora se nota quando N=1, diminuindo o ruído drasticamente, com pouca perda de detalhe. Quando N é alto, como quando N=11, não há muitas mudanças.

Agora variando M e mantendo N=3:

- Média:

<img src="C:\Users\Marcos\Desktop\facul\visao\stereoAssignment\dispMeanAgg_M_Comparison.png" alt="dispMeanAgg_M_Comparison" style="zoom:50%;" />

- Mediana:

<img src="C:\Users\Marcos\Desktop\facul\visao\stereoAssignment\dispMedianAgg_M_Comparison.png" alt="dispMedianAgg_M_Comparison" style="zoom:50%;" />

Nota-se uma redução no ruído e redução de detalhes com o aumento de M, assim como com o aumento de N.

Alguns artefatos são comuns em todos os resultados:

- Faixa com ruído à esquerda: Isso se dá pois o deslocamento é calculado da imagem que foi tirada mais à esquerda para a que foi tirada mais à direita, então uma parte da imagem da esquerda não aparece na imagem da direita.
- Ruído à esquerda das bordas dos objetos: De forma semelhante, algumas partes do cenário estão escondidas na imagem da direita, então falta informação para detectar o deslocamento dos pixels correspondentes a estas partes do cenário