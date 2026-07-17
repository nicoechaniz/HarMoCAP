# Bitacora - HarMoCAP

> Log cronologico del proyecto. La edicion canonica corresponde al rol auditor activo o a una instruccion explicita del usuario.

---

## 2026-07-17 - S0 - Init del workspace

Workspace inicializado con `mandinga_init_workspace` en `/mnt/m2-1TB/HarMoCAP`.

Definicion inicial: desarrollo de software para pose estimation y tracking corporal con Ultralytics YOLO26m-pose. El primer workflow usara COCO8-pose como dataset pequeno de validacion tecnica; el objetivo posterior es seleccionar y curar datasets mas grandes para tracking en tiempo real de una o varias personas.

Integracion prevista: las variables de movimiento alimentaran un modulador en tiempo real de la armonia de Harmonic Beacon. La percepcion, el tracking, la representacion corporal y la modulacion se mantienen como capas separables.

Roles: Claude y Codex pueden actuar como implementador o auditor segun la asignacion casuistica del usuario; no hay ownership permanente.

Archivos sembrados:

- `AGENTS.md` + `CLAUDE.md`
- `BITACORA.md`
- `PENDIENTES.md`
- `NOTAS_CLAUDE-CODEX.md` + `NOTAS_CODEX-CLAUDE.md`
- `README.md`
- `.gitignore`
- `Biblioteca/INDEX.md`

Memoria persistente de Codex: `/root/.codex/memories/mariano_global_directives_2026-07-09.md`. Las 25 directivas universales de `mandinga_init_workspace` ya constan alli; no se duplicaron.

Mensaje recursivo `001` integrado: se adopta el uso activo de la memoria colectiva y el registro de notas nuevas en `/mnt/m2-1TB/inbox/new/` cuando un hallazgo sea relevante para otros proyectos. El mapa permanece read-only.

La documentacion tecnica de pose consultada es `https://docs.ultralytics.com/tasks/pose#train`; el contexto conceptual inicial se ancla en HIT cap. 12 y el marco Beacon-HIT.

Repositorio principal: `Mar-IA-no/HarMoCAP` (privado), creado y publicado con los commits `a28bd8c` y `66b5004`. Espejo privado previsto: `AlterMundi/HarMoCAP`. El remoto local `altermundi` y los dos `pushurl` de `origin` ya estan configurados; falta que un administrador cree el repositorio organizacional porque la cuenta autenticada no tiene el scope/permisos necesarios para hacerlo.

## 2026-07-17 - S1 - Visibilidad y espejo GitHub

Por decision del usuario, ambos repositorios quedaron publicos:

- `Mar-IA-no/HarMoCAP`
- `AlterMundi/HarMoCAP`

El repo organizacional fue creado y el personal cambio de privado a publico. El push dual local queda habilitado para sincronizar `main` en cada `git push origin`.

## 2026-07-17 - S2 - Mensajes recursivos 002-003 integrados

Los mensajes recursivos `002` y `003` agregaron el contexto operativo de GitHub:

- `Mar-IA-no` autentica mediante el token OAuth de `gh`.
- `AlterMundi` usa un PAT fine-grained separado, con `Contents R/W` solo sobre repositorios autorizados.
- Crear un repositorio organizacional y pushear a uno existente son operaciones distintas.
- El repositorio nuevo `AlterMundi/HarMoCAP` ya existe y es publico, pero el PAT de AlterMundi aun debe incluirlo en su allowlist para que el push automatico funcione.

La prueba de `git push origin main` publico `a124285` en `Mar-IA-no` y recibio `403` en `AlterMundi`; el mismo commit fue publicado manualmente en el espejo con la credencial de `gh`. Por lo tanto, ambos `main` estan alineados en `a124285`, pero la sincronizacion automatica queda condicionada a que un owner agregue `AlterMundi/HarMoCAP` al PAT fine-grained. No se copiaron tokens a archivos del proyecto ni a la memoria colectiva.

## 2026-07-17 - S3 - Sincronizacion GitHub verificada

El usuario actualizo el PAT fine-grained de AlterMundi para incluir `HarMoCAP` con permiso de escritura. La prueba posterior de `git push origin main` termino correctamente para los dos destinos:

- `Mar-IA-no/HarMoCAP`: publico, `main` en `a1242853cd6df0e3a0771c1edcc5fbe35a605931`.
- `AlterMundi/HarMoCAP`: publico, `main` en `a1242853cd6df0e3a0771c1edcc5fbe35a605931`.

La sincronizacion automatica del espejo queda operativa mediante los dos `pushurl` de `origin`.
