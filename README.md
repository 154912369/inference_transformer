# inference_transformer
$$
C + i*{strideC} = \alpha\text{op}(A + i*{strideA})\text{op}(B + i*{strideB}) + \beta(C + i*{strideC}),\text{ for i } \in \lbrack 0,batchCount - 1\rbrack
$$

$$
\text{op}(A) = \left\{ \begin{matrix} A & {\text{if }\textsf{transa == $\mathrm{CUBLAS\_OP\_N}$}} \\ A^{T} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\_OP\_T}$}} \\ A^{H} & {\text{if }\textsf{transa == $\mathrm{CUBLAS\_OP\_C}$}} \\ \end{matrix} \right.
$$