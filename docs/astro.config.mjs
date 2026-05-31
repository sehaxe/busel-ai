import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: 'Busel AI',
			social: [
				// Новый синтаксис соцсетей для Starlight v0.33.0+ (массив вместо объекта)
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/sehaxe/busel-ai' }
			],
			sidebar: [
				{
					label: 'Guides',
					items: [
						'guides/getting-started',
					],
				},
			],
		}),
	],
});