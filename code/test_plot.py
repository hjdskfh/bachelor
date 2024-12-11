import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [4, 5, 6], label='Example')
plt.xlabel(r'$\Delta I \, (\mathrm{mA})$')  # ΔI (mA)
plt.ylabel(r'$\Delta \langle \mu \rangle$')  # Δ⟨μ⟩
plt.legend()
plt.show()
