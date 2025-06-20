import { loadEnv } from "vite";
import { defineConfig } from 'astro/config';

import expressiveCode from 'astro-expressive-code';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import spectre from './package/src';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypeExpressiveCode from 'rehype-expressive-code'

/** @type {import('rehype-expressive-code').RehypeExpressiveCodeOptions} */
const rehypeExpressiveCodeOptions = {
  themes: ['dracula'], 
}

import node from '@astrojs/node';
import { spectreDark } from './src/ec-theme';

// const {
//   GISCUS_REPO,
//   GISCUS_REPO_ID,
//   GISCUS_CATEGORY,
//   GISCUS_CATEGORY_ID,
//   GISCUS_MAPPING,
//   GISCUS_STRICT,
//   GISCUS_REACTIONS_ENABLED,
//   GISCUS_EMIT_METADATA,
//   GISCUS_LANG
// } = loadEnv(process.env.NODE_ENV!, process.cwd(), "");

// https://astro.build/config
const config = defineConfig({
  site: 'https://kkaryl.github.io',
  output: 'static',
  integrations: [
    expressiveCode({
      themes: ['dracula'], //spectreDark
    }),
    mdx({
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        rehypeKatex,
        [rehypeExpressiveCode, rehypeExpressiveCodeOptions],
      ],
    }),
    sitemap(),
    spectre({
      name: 'Karyl Ong',
      openGraph: {
        home: {
          title: 'Karyl Ong - Build to Learn',
          description: 'About me.'
        },
        blog: {
          title: 'Blog',
          description: 'Short posts to share my thoughts.'
        },
        projects: {
          title: 'Projects'
        }
      },
      // giscus: {
      //   repository: GISCUS_REPO,
      //   repositoryId: GISCUS_REPO_ID,
      //   category: GISCUS_CATEGORY,
      //   categoryId: GISCUS_CATEGORY_ID,
      //   mapping: GISCUS_MAPPING as any,
      //   strict: GISCUS_STRICT === "true",
      //   reactionsEnabled: GISCUS_REACTIONS_ENABLED === "true",
      //   emitMetadata: GISCUS_EMIT_METADATA === "true",
      //   lang: GISCUS_LANG,
      // }
    })
  ],
  // adapter: node({
  //   mode: 'standalone'
  // })
});

export default config;