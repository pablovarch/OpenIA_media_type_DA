import ssl_analyzer_da
import openai_media_type_dom_atr
import dom_att_classifier_v2
import asyncio

if __name__ == '__main__':
    print("Iniciando openai_media_type_dom_atr...")
    asyncio.run(openai_media_type_dom_atr.main())
    print("✓ Completado openai_media_type_dom_atr\n")

    print("Iniciando ssl_analyzer_da...")
    asyncio.run(ssl_analyzer_da.run_backfill())
    print("✓ Completado ssl_analyzer_da\n")

    print("Iniciando dom_att_classifier_v2...")
    asyncio.run(dom_att_classifier_v2.main())
    print("✓ Completado dom_att_classifier_v2\n")

    print("Todos los procesos finalizados")
