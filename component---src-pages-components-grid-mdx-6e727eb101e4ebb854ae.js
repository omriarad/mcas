(window.webpackJsonp=window.webpackJsonp||[]).push([[28],{"013z":function(e,t,a){"use strict";var n=a("q1tI"),c=a.n(n),b=a("NmYn"),l=a.n(b),r=a("Wbzz"),i=a("Xrax"),m=a("k4MR"),o=a("TSYQ"),s=a.n(o),d=a("QH2O"),p=a.n(d),g=a("qKvR"),u=function(e){var t,a=e.title,n=e.theme,c=e.tabs,b=void 0===c?[]:c;return Object(g.b)("div",{className:s()(p.a.pageHeader,(t={},t[p.a.withTabs]=b.length,t[p.a.darkMode]="dark"===n,t))},Object(g.b)("div",{className:"bx--grid"},Object(g.b)("div",{className:"bx--row"},Object(g.b)("div",{className:"bx--col-lg-12"},Object(g.b)("h1",{id:"page-title",className:p.a.text},a)))))},A=a("BAC9"),O=function(e){var t=e.relativePagePath,a=e.repository,n=Object(r.useStaticQuery)("1364590287").site.siteMetadata.repository,c=a||n,b=c.baseUrl,l=c.subDirectory,i=b+"/edit/"+c.branch+l+"/src/pages"+t;return b?Object(g.b)("div",{className:"bx--row "+A.row},Object(g.b)("div",{className:"bx--col"},Object(g.b)("a",{className:A.link,href:i},"Edit this page on GitHub"))):null},j=a("FCXl"),N=a("dI71"),y=a("I8xM"),h=function(e){function t(){return e.apply(this,arguments)||this}return Object(N.a)(t,e),t.prototype.render=function(){var e=this.props,t=e.title,a=e.tabs,n=e.slug,c=n.split("/").filter(Boolean).slice(-1)[0],b=a.map((function(e){var t,a=l()(e,{lower:!0,strict:!0}),b=a===c,i=new RegExp(c+"/?(#.*)?$"),m=n.replace(i,a);return Object(g.b)("li",{key:e,className:s()((t={},t[y.selectedItem]=b,t),y.listItem)},Object(g.b)(r.Link,{className:y.link,to:""+m},e))}));return Object(g.b)("div",{className:y.tabsContainer},Object(g.b)("div",{className:"bx--grid"},Object(g.b)("div",{className:"bx--row"},Object(g.b)("div",{className:"bx--col-lg-12 bx--col-no-gutter"},Object(g.b)("nav",{"aria-label":t},Object(g.b)("ul",{className:y.list},b))))))},t}(c.a.Component),k=a("MjG9"),x=a("CzIb");t.a=function(e){var t=e.pageContext,a=e.children,n=e.location,c=e.Title,b=t.frontmatter,o=void 0===b?{}:b,s=t.relativePagePath,d=t.titleType,p=o.tabs,A=o.title,N=o.theme,y=o.description,f=o.keywords,E=Object(x.a)().interiorTheme,w=Object(r.useStaticQuery)("2456312558").site.pathPrefix,R=w?n.pathname.replace(w,""):n.pathname,K=p?R.split("/").filter(Boolean).slice(-1)[0]||l()(p[0],{lower:!0}):"",S=N||E;return Object(g.b)(m.a,{tabs:p,homepage:!1,theme:S,pageTitle:A,pageDescription:y,pageKeywords:f,titleType:d},Object(g.b)(u,{title:c?Object(g.b)(c,null):A,label:"label",tabs:p,theme:S}),p&&Object(g.b)(h,{title:A,slug:R,tabs:p,currentTab:K}),Object(g.b)(k.a,{padded:!0},a,Object(g.b)(O,{relativePagePath:s})),Object(g.b)(j.a,{pageContext:t,location:n,slug:R,tabs:p,currentTab:K}),Object(g.b)(i.a,null))}},BAC9:function(e,t,a){e.exports={bxTextTruncateEnd:"EditLink-module--bx--text-truncate--end--2pqje",bxTextTruncateFront:"EditLink-module--bx--text-truncate--front--3_lIE",link:"EditLink-module--link--1qzW3",row:"EditLink-module--row--1B9Gk"}},I8xM:function(e,t,a){e.exports={bxTextTruncateEnd:"PageTabs-module--bx--text-truncate--end--267NA",bxTextTruncateFront:"PageTabs-module--bx--text-truncate--front--3xEQF",tabsContainer:"PageTabs-module--tabs-container--8N4k0",list:"PageTabs-module--list--3eFQc",listItem:"PageTabs-module--list-item--nUmtD",link:"PageTabs-module--link--1mDJ1",selectedItem:"PageTabs-module--selected-item--YPVr3"}},"J3+z":function(e,t,a){"use strict";a.r(t),a.d(t,"_frontmatter",(function(){return r})),a.d(t,"default",(function(){return u}));var n=a("wx14"),c=a("zLVn"),b=(a("q1tI"),a("7ljp")),l=a("013z"),r=(a("qKvR"),{}),i=function(e){return function(t){return console.warn("Component "+e+" was not imported, exported, or provided by MDXProvider as global scope"),Object(b.b)("div",t)}},m=i("PageDescription"),o=i("Title"),s=i("Row"),d=i("Column"),p={_frontmatter:r},g=l.a;function u(e){var t=e.components,a=Object(c.a)(e,["components"]);return Object(b.b)(g,Object(n.a)({},p,a,{components:t,mdxType:"MDXLayout"}),Object(b.b)(m,{mdxType:"PageDescription"},Object(b.b)("p",null,Object(b.b)("inlineCode",{parentName:"p"},"<Row>")," and ",Object(b.b)("inlineCode",{parentName:"p"},"<Column>")," components are used to arrange content and components on the grid within a page.\nTo learn more about the grid is built, you can read the docs in the ",Object(b.b)("a",{parentName:"p",href:"https://github.com/carbon-design-system/carbon/tree/master/packages/grid"},"Carbon")," repo.")),Object(b.b)("h2",null,"Row"),Object(b.b)("p",null,"The ",Object(b.b)("inlineCode",{parentName:"p"},"<Row>")," component is a wrapper that adds the ",Object(b.b)("inlineCode",{parentName:"p"},"bx--row")," class to a wrapper div. You will want to use this to define rows that you will place ",Object(b.b)("inlineCode",{parentName:"p"},"<Column>")," components inside of."),Object(b.b)("h3",null,"Code"),Object(b.b)("pre",null,Object(b.b)("code",{parentName:"pre",className:"language-jsx",metastring:"path=components/Grid.js src=https://github.com/carbon-design-system/gatsby-theme-carbon/tree/master/packages/gatsby-theme-carbon/src/components/Grid",path:"components/Grid.js",src:"https://github.com/carbon-design-system/gatsby-theme-carbon/tree/master/packages/gatsby-theme-carbon/src/components/Grid"},"<Row>\n  <Column>\n    Content or additional <Components />\n  </Column>\n</Row>\n")),Object(b.b)(o,{mdxType:"Title"},"Row props"),Object(b.b)("table",null,Object(b.b)("thead",{parentName:"table"},Object(b.b)("tr",{parentName:"thead"},Object(b.b)("th",{parentName:"tr",align:null},"property"),Object(b.b)("th",{parentName:"tr",align:null},"propType"),Object(b.b)("th",{parentName:"tr",align:null},"required"),Object(b.b)("th",{parentName:"tr",align:null},"default"),Object(b.b)("th",{parentName:"tr",align:null},"description"))),Object(b.b)("tbody",{parentName:"table"},Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"children"),Object(b.b)("td",{parentName:"tr",align:null},"node"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null})),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"className"),Object(b.b)("td",{parentName:"tr",align:null},"string"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Add custom class name")))),Object(b.b)("h2",null,"Column"),Object(b.b)("p",null,"The ",Object(b.b)("inlineCode",{parentName:"p"},"<Column>")," component is used to define column widths for your content, you can set the rules at different breakpoints with the props."),Object(b.b)("h3",null,"Example"),Object(b.b)(s,{mdxType:"Row"},Object(b.b)(d,{colMd:4,colLg:4,mdxType:"Column"},Object(b.b)("span",{className:"gatsby-resp-image-wrapper",style:{position:"relative",display:"block",marginLeft:"auto",marginRight:"auto",maxWidth:"1152px"}},"\n      ",Object(b.b)("span",{parentName:"span",className:"gatsby-resp-image-background-image",style:{paddingBottom:"56.25%",position:"relative",bottom:"0",left:"0",backgroundImage:"url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAALCAYAAAB/Ca1DAAAACXBIWXMAAAsTAAALEwEAmpwYAAACE0lEQVQoz1WSS2/aQBRGs2DdKsFj4xcBE5FSERahjfFzPDavmqpqEFLaZUMbKrFmgUT/VaUu+uO+3hkgJYujgYs5/ubee8aYhnrzGu0bH1rNBTMvoVsNQp5NaM4bWLaDKAyQxLEiiiIUeY6vXx4wn9+rWnzgjDGGRqOB3k0XjuPCMAxUq1VomqaoVi9gmibihIOLERKeEQJ5Mcb9fIFPnxdIklTJkiTZCz3PQ6/XQ6fTQbvdVt8VrRaazSYsyyRJgWxY7qUpJzKkYgxelPT5RChTSElRDMGzDELkyOk6QgiqFeCcU3KbEmbqj1LE8/FeTIKU5+oFJ0KGq5YHwWNkaaSQEol8IAhDODYJSSSvpoQk+y8ULxPKoTS8NvrRBLfBSPHuvY9+v6966zgObMtCmg0hRh9VupT6mGY5smKKjGovhDXtFc6vZ6iUf1H58AeV6W+c12/Bqq/BdF0NyKAzCGmKqUAYxQjjFEGUIKLzmPRZKB/Waw6Y2yHeKoyaRdPWwZiObrer+vl9ucTq6QeeiJ+rFdbrNZZUK3Kh1ugoJSGDbtb3u2dYJLfA7BZ026O1ucDdnY/FYoHtdovdbkf8wmazwbfHR0wmk+dkRw7CS5JcKZlu1PZC+QL6zXVdtQXT6RRlWWI2m6kt8H0fIQ3smOzkyoeEUmR5BMkMU9VkO+RayWUfDAZKEASBumIqV+ikd0f+AcBmYOtXGEYsAAAAAElFTkSuQmCC')",backgroundSize:"cover",display:"block"}}),"\n  ",Object(b.b)("img",{parentName:"span",className:"gatsby-resp-image-image",alt:"Grid Example",title:"Grid Example",src:"/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png",srcSet:["/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/7fc1e/Article_05.png 288w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/a5df1/Article_05.png 576w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png 1152w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/362ee/Article_05.png 1600w"],sizes:"(max-width: 1152px) 100vw, 1152px",style:{width:"100%",height:"100%",margin:"0",verticalAlign:"middle",position:"absolute",top:"0",left:"0"},loading:"lazy"}),"\n    ")),Object(b.b)(d,{colMd:4,colLg:4,mdxType:"Column"},Object(b.b)("span",{className:"gatsby-resp-image-wrapper",style:{position:"relative",display:"block",marginLeft:"auto",marginRight:"auto",maxWidth:"1152px"}},"\n      ",Object(b.b)("span",{parentName:"span",className:"gatsby-resp-image-background-image",style:{paddingBottom:"56.25%",position:"relative",bottom:"0",left:"0",backgroundImage:"url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAALCAYAAAB/Ca1DAAAACXBIWXMAAAsTAAALEwEAmpwYAAACE0lEQVQoz1WSS2/aQBRGs2DdKsFj4xcBE5FSERahjfFzPDavmqpqEFLaZUMbKrFmgUT/VaUu+uO+3hkgJYujgYs5/ubee8aYhnrzGu0bH1rNBTMvoVsNQp5NaM4bWLaDKAyQxLEiiiIUeY6vXx4wn9+rWnzgjDGGRqOB3k0XjuPCMAxUq1VomqaoVi9gmibihIOLERKeEQJ5Mcb9fIFPnxdIklTJkiTZCz3PQ6/XQ6fTQbvdVt8VrRaazSYsyyRJgWxY7qUpJzKkYgxelPT5RChTSElRDMGzDELkyOk6QgiqFeCcU3KbEmbqj1LE8/FeTIKU5+oFJ0KGq5YHwWNkaaSQEol8IAhDODYJSSSvpoQk+y8ULxPKoTS8NvrRBLfBSPHuvY9+v6966zgObMtCmg0hRh9VupT6mGY5smKKjGovhDXtFc6vZ6iUf1H58AeV6W+c12/Bqq/BdF0NyKAzCGmKqUAYxQjjFEGUIKLzmPRZKB/Waw6Y2yHeKoyaRdPWwZiObrer+vl9ucTq6QeeiJ+rFdbrNZZUK3Kh1ugoJSGDbtb3u2dYJLfA7BZ026O1ucDdnY/FYoHtdovdbkf8wmazwbfHR0wmk+dkRw7CS5JcKZlu1PZC+QL6zXVdtQXT6RRlWWI2m6kt8H0fIQ3smOzkyoeEUmR5BMkMU9VkO+RayWUfDAZKEASBumIqV+ikd0f+AcBmYOtXGEYsAAAAAElFTkSuQmCC')",backgroundSize:"cover",display:"block"}}),"\n  ",Object(b.b)("img",{parentName:"span",className:"gatsby-resp-image-image",alt:"Grid Example",title:"Grid Example",src:"/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png",srcSet:["/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/7fc1e/Article_05.png 288w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/a5df1/Article_05.png 576w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png 1152w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/362ee/Article_05.png 1600w"],sizes:"(max-width: 1152px) 100vw, 1152px",style:{width:"100%",height:"100%",margin:"0",verticalAlign:"middle",position:"absolute",top:"0",left:"0"},loading:"lazy"}),"\n    ")),Object(b.b)(d,{colMd:4,colLg:4,mdxType:"Column"},Object(b.b)("span",{className:"gatsby-resp-image-wrapper",style:{position:"relative",display:"block",marginLeft:"auto",marginRight:"auto",maxWidth:"1152px"}},"\n      ",Object(b.b)("span",{parentName:"span",className:"gatsby-resp-image-background-image",style:{paddingBottom:"56.25%",position:"relative",bottom:"0",left:"0",backgroundImage:"url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAALCAYAAAB/Ca1DAAAACXBIWXMAAAsTAAALEwEAmpwYAAACE0lEQVQoz1WSS2/aQBRGs2DdKsFj4xcBE5FSERahjfFzPDavmqpqEFLaZUMbKrFmgUT/VaUu+uO+3hkgJYujgYs5/ubee8aYhnrzGu0bH1rNBTMvoVsNQp5NaM4bWLaDKAyQxLEiiiIUeY6vXx4wn9+rWnzgjDGGRqOB3k0XjuPCMAxUq1VomqaoVi9gmibihIOLERKeEQJ5Mcb9fIFPnxdIklTJkiTZCz3PQ6/XQ6fTQbvdVt8VrRaazSYsyyRJgWxY7qUpJzKkYgxelPT5RChTSElRDMGzDELkyOk6QgiqFeCcU3KbEmbqj1LE8/FeTIKU5+oFJ0KGq5YHwWNkaaSQEol8IAhDODYJSSSvpoQk+y8ULxPKoTS8NvrRBLfBSPHuvY9+v6966zgObMtCmg0hRh9VupT6mGY5smKKjGovhDXtFc6vZ6iUf1H58AeV6W+c12/Bqq/BdF0NyKAzCGmKqUAYxQjjFEGUIKLzmPRZKB/Waw6Y2yHeKoyaRdPWwZiObrer+vl9ucTq6QeeiJ+rFdbrNZZUK3Kh1ugoJSGDbtb3u2dYJLfA7BZ026O1ucDdnY/FYoHtdovdbkf8wmazwbfHR0wmk+dkRw7CS5JcKZlu1PZC+QL6zXVdtQXT6RRlWWI2m6kt8H0fIQ3smOzkyoeEUmR5BMkMU9VkO+RayWUfDAZKEASBumIqV+ikd0f+AcBmYOtXGEYsAAAAAElFTkSuQmCC')",backgroundSize:"cover",display:"block"}}),"\n  ",Object(b.b)("img",{parentName:"span",className:"gatsby-resp-image-image",alt:"Grid Example",title:"Grid Example",src:"/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png",srcSet:["/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/7fc1e/Article_05.png 288w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/a5df1/Article_05.png 576w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png 1152w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/362ee/Article_05.png 1600w"],sizes:"(max-width: 1152px) 100vw, 1152px",style:{width:"100%",height:"100%",margin:"0",verticalAlign:"middle",position:"absolute",top:"0",left:"0"},loading:"lazy"}),"\n    "))),Object(b.b)(o,{mdxType:"Title"},"No gutter left"),Object(b.b)(s,{mdxType:"Row"},Object(b.b)(d,{colMd:4,colLg:4,noGutterMdLeft:!0,mdxType:"Column"},Object(b.b)("span",{className:"gatsby-resp-image-wrapper",style:{position:"relative",display:"block",marginLeft:"auto",marginRight:"auto",maxWidth:"1152px"}},"\n      ",Object(b.b)("span",{parentName:"span",className:"gatsby-resp-image-background-image",style:{paddingBottom:"56.25%",position:"relative",bottom:"0",left:"0",backgroundImage:"url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAALCAYAAAB/Ca1DAAAACXBIWXMAAAsTAAALEwEAmpwYAAACE0lEQVQoz1WSS2/aQBRGs2DdKsFj4xcBE5FSERahjfFzPDavmqpqEFLaZUMbKrFmgUT/VaUu+uO+3hkgJYujgYs5/ubee8aYhnrzGu0bH1rNBTMvoVsNQp5NaM4bWLaDKAyQxLEiiiIUeY6vXx4wn9+rWnzgjDGGRqOB3k0XjuPCMAxUq1VomqaoVi9gmibihIOLERKeEQJ5Mcb9fIFPnxdIklTJkiTZCz3PQ6/XQ6fTQbvdVt8VrRaazSYsyyRJgWxY7qUpJzKkYgxelPT5RChTSElRDMGzDELkyOk6QgiqFeCcU3KbEmbqj1LE8/FeTIKU5+oFJ0KGq5YHwWNkaaSQEol8IAhDODYJSSSvpoQk+y8ULxPKoTS8NvrRBLfBSPHuvY9+v6966zgObMtCmg0hRh9VupT6mGY5smKKjGovhDXtFc6vZ6iUf1H58AeV6W+c12/Bqq/BdF0NyKAzCGmKqUAYxQjjFEGUIKLzmPRZKB/Waw6Y2yHeKoyaRdPWwZiObrer+vl9ucTq6QeeiJ+rFdbrNZZUK3Kh1ugoJSGDbtb3u2dYJLfA7BZ026O1ucDdnY/FYoHtdovdbkf8wmazwbfHR0wmk+dkRw7CS5JcKZlu1PZC+QL6zXVdtQXT6RRlWWI2m6kt8H0fIQ3smOzkyoeEUmR5BMkMU9VkO+RayWUfDAZKEASBumIqV+ikd0f+AcBmYOtXGEYsAAAAAElFTkSuQmCC')",backgroundSize:"cover",display:"block"}}),"\n  ",Object(b.b)("img",{parentName:"span",className:"gatsby-resp-image-image",alt:"Grid Example",title:"Grid Example",src:"/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png",srcSet:["/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/7fc1e/Article_05.png 288w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/a5df1/Article_05.png 576w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png 1152w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/362ee/Article_05.png 1600w"],sizes:"(max-width: 1152px) 100vw, 1152px",style:{width:"100%",height:"100%",margin:"0",verticalAlign:"middle",position:"absolute",top:"0",left:"0"},loading:"lazy"}),"\n    ")),Object(b.b)(d,{colMd:4,colLg:4,noGutterMdLeft:!0,mdxType:"Column"},Object(b.b)("span",{className:"gatsby-resp-image-wrapper",style:{position:"relative",display:"block",marginLeft:"auto",marginRight:"auto",maxWidth:"1152px"}},"\n      ",Object(b.b)("span",{parentName:"span",className:"gatsby-resp-image-background-image",style:{paddingBottom:"56.25%",position:"relative",bottom:"0",left:"0",backgroundImage:"url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAALCAYAAAB/Ca1DAAAACXBIWXMAAAsTAAALEwEAmpwYAAACE0lEQVQoz1WSS2/aQBRGs2DdKsFj4xcBE5FSERahjfFzPDavmqpqEFLaZUMbKrFmgUT/VaUu+uO+3hkgJYujgYs5/ubee8aYhnrzGu0bH1rNBTMvoVsNQp5NaM4bWLaDKAyQxLEiiiIUeY6vXx4wn9+rWnzgjDGGRqOB3k0XjuPCMAxUq1VomqaoVi9gmibihIOLERKeEQJ5Mcb9fIFPnxdIklTJkiTZCz3PQ6/XQ6fTQbvdVt8VrRaazSYsyyRJgWxY7qUpJzKkYgxelPT5RChTSElRDMGzDELkyOk6QgiqFeCcU3KbEmbqj1LE8/FeTIKU5+oFJ0KGq5YHwWNkaaSQEol8IAhDODYJSSSvpoQk+y8ULxPKoTS8NvrRBLfBSPHuvY9+v6966zgObMtCmg0hRh9VupT6mGY5smKKjGovhDXtFc6vZ6iUf1H58AeV6W+c12/Bqq/BdF0NyKAzCGmKqUAYxQjjFEGUIKLzmPRZKB/Waw6Y2yHeKoyaRdPWwZiObrer+vl9ucTq6QeeiJ+rFdbrNZZUK3Kh1ugoJSGDbtb3u2dYJLfA7BZ026O1ucDdnY/FYoHtdovdbkf8wmazwbfHR0wmk+dkRw7CS5JcKZlu1PZC+QL6zXVdtQXT6RRlWWI2m6kt8H0fIQ3smOzkyoeEUmR5BMkMU9VkO+RayWUfDAZKEASBumIqV+ikd0f+AcBmYOtXGEYsAAAAAElFTkSuQmCC')",backgroundSize:"cover",display:"block"}}),"\n  ",Object(b.b)("img",{parentName:"span",className:"gatsby-resp-image-image",alt:"Grid Example",title:"Grid Example",src:"/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png",srcSet:["/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/7fc1e/Article_05.png 288w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/a5df1/Article_05.png 576w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png 1152w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/362ee/Article_05.png 1600w"],sizes:"(max-width: 1152px) 100vw, 1152px",style:{width:"100%",height:"100%",margin:"0",verticalAlign:"middle",position:"absolute",top:"0",left:"0"},loading:"lazy"}),"\n    ")),Object(b.b)(d,{colMd:4,colLg:4,noGutterMdLeft:!0,mdxType:"Column"},Object(b.b)("span",{className:"gatsby-resp-image-wrapper",style:{position:"relative",display:"block",marginLeft:"auto",marginRight:"auto",maxWidth:"1152px"}},"\n      ",Object(b.b)("span",{parentName:"span",className:"gatsby-resp-image-background-image",style:{paddingBottom:"56.25%",position:"relative",bottom:"0",left:"0",backgroundImage:"url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAALCAYAAAB/Ca1DAAAACXBIWXMAAAsTAAALEwEAmpwYAAACE0lEQVQoz1WSS2/aQBRGs2DdKsFj4xcBE5FSERahjfFzPDavmqpqEFLaZUMbKrFmgUT/VaUu+uO+3hkgJYujgYs5/ubee8aYhnrzGu0bH1rNBTMvoVsNQp5NaM4bWLaDKAyQxLEiiiIUeY6vXx4wn9+rWnzgjDGGRqOB3k0XjuPCMAxUq1VomqaoVi9gmibihIOLERKeEQJ5Mcb9fIFPnxdIklTJkiTZCz3PQ6/XQ6fTQbvdVt8VrRaazSYsyyRJgWxY7qUpJzKkYgxelPT5RChTSElRDMGzDELkyOk6QgiqFeCcU3KbEmbqj1LE8/FeTIKU5+oFJ0KGq5YHwWNkaaSQEol8IAhDODYJSSSvpoQk+y8ULxPKoTS8NvrRBLfBSPHuvY9+v6966zgObMtCmg0hRh9VupT6mGY5smKKjGovhDXtFc6vZ6iUf1H58AeV6W+c12/Bqq/BdF0NyKAzCGmKqUAYxQjjFEGUIKLzmPRZKB/Waw6Y2yHeKoyaRdPWwZiObrer+vl9ucTq6QeeiJ+rFdbrNZZUK3Kh1ugoJSGDbtb3u2dYJLfA7BZ026O1ucDdnY/FYoHtdovdbkf8wmazwbfHR0wmk+dkRw7CS5JcKZlu1PZC+QL6zXVdtQXT6RRlWWI2m6kt8H0fIQ3smOzkyoeEUmR5BMkMU9VkO+RayWUfDAZKEASBumIqV+ikd0f+AcBmYOtXGEYsAAAAAElFTkSuQmCC')",backgroundSize:"cover",display:"block"}}),"\n  ",Object(b.b)("img",{parentName:"span",className:"gatsby-resp-image-image",alt:"Grid Example",title:"Grid Example",src:"/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png",srcSet:["/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/7fc1e/Article_05.png 288w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/a5df1/Article_05.png 576w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png 1152w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/362ee/Article_05.png 1600w"],sizes:"(max-width: 1152px) 100vw, 1152px",style:{width:"100%",height:"100%",margin:"0",verticalAlign:"middle",position:"absolute",top:"0",left:"0"},loading:"lazy"}),"\n    "))),Object(b.b)(o,{mdxType:"Title"},"No gutter"),Object(b.b)(s,{mdxType:"Row"},Object(b.b)(d,{colMd:4,colLg:4,noGutterSm:!0,mdxType:"Column"},Object(b.b)("span",{className:"gatsby-resp-image-wrapper",style:{position:"relative",display:"block",marginLeft:"auto",marginRight:"auto",maxWidth:"1152px"}},"\n      ",Object(b.b)("span",{parentName:"span",className:"gatsby-resp-image-background-image",style:{paddingBottom:"56.25%",position:"relative",bottom:"0",left:"0",backgroundImage:"url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAALCAYAAAB/Ca1DAAAACXBIWXMAAAsTAAALEwEAmpwYAAACE0lEQVQoz1WSS2/aQBRGs2DdKsFj4xcBE5FSERahjfFzPDavmqpqEFLaZUMbKrFmgUT/VaUu+uO+3hkgJYujgYs5/ubee8aYhnrzGu0bH1rNBTMvoVsNQp5NaM4bWLaDKAyQxLEiiiIUeY6vXx4wn9+rWnzgjDGGRqOB3k0XjuPCMAxUq1VomqaoVi9gmibihIOLERKeEQJ5Mcb9fIFPnxdIklTJkiTZCz3PQ6/XQ6fTQbvdVt8VrRaazSYsyyRJgWxY7qUpJzKkYgxelPT5RChTSElRDMGzDELkyOk6QgiqFeCcU3KbEmbqj1LE8/FeTIKU5+oFJ0KGq5YHwWNkaaSQEol8IAhDODYJSSSvpoQk+y8ULxPKoTS8NvrRBLfBSPHuvY9+v6966zgObMtCmg0hRh9VupT6mGY5smKKjGovhDXtFc6vZ6iUf1H58AeV6W+c12/Bqq/BdF0NyKAzCGmKqUAYxQjjFEGUIKLzmPRZKB/Waw6Y2yHeKoyaRdPWwZiObrer+vl9ucTq6QeeiJ+rFdbrNZZUK3Kh1ugoJSGDbtb3u2dYJLfA7BZ026O1ucDdnY/FYoHtdovdbkf8wmazwbfHR0wmk+dkRw7CS5JcKZlu1PZC+QL6zXVdtQXT6RRlWWI2m6kt8H0fIQ3smOzkyoeEUmR5BMkMU9VkO+RayWUfDAZKEASBumIqV+ikd0f+AcBmYOtXGEYsAAAAAElFTkSuQmCC')",backgroundSize:"cover",display:"block"}}),"\n  ",Object(b.b)("img",{parentName:"span",className:"gatsby-resp-image-image",alt:"Grid Example",title:"Grid Example",src:"/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png",srcSet:["/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/7fc1e/Article_05.png 288w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/a5df1/Article_05.png 576w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png 1152w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/362ee/Article_05.png 1600w"],sizes:"(max-width: 1152px) 100vw, 1152px",style:{width:"100%",height:"100%",margin:"0",verticalAlign:"middle",position:"absolute",top:"0",left:"0"},loading:"lazy"}),"\n    ")),Object(b.b)(d,{colMd:4,colLg:4,noGutterSm:!0,mdxType:"Column"},Object(b.b)("span",{className:"gatsby-resp-image-wrapper",style:{position:"relative",display:"block",marginLeft:"auto",marginRight:"auto",maxWidth:"1152px"}},"\n      ",Object(b.b)("span",{parentName:"span",className:"gatsby-resp-image-background-image",style:{paddingBottom:"56.25%",position:"relative",bottom:"0",left:"0",backgroundImage:"url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAALCAYAAAB/Ca1DAAAACXBIWXMAAAsTAAALEwEAmpwYAAACE0lEQVQoz1WSS2/aQBRGs2DdKsFj4xcBE5FSERahjfFzPDavmqpqEFLaZUMbKrFmgUT/VaUu+uO+3hkgJYujgYs5/ubee8aYhnrzGu0bH1rNBTMvoVsNQp5NaM4bWLaDKAyQxLEiiiIUeY6vXx4wn9+rWnzgjDGGRqOB3k0XjuPCMAxUq1VomqaoVi9gmibihIOLERKeEQJ5Mcb9fIFPnxdIklTJkiTZCz3PQ6/XQ6fTQbvdVt8VrRaazSYsyyRJgWxY7qUpJzKkYgxelPT5RChTSElRDMGzDELkyOk6QgiqFeCcU3KbEmbqj1LE8/FeTIKU5+oFJ0KGq5YHwWNkaaSQEol8IAhDODYJSSSvpoQk+y8ULxPKoTS8NvrRBLfBSPHuvY9+v6966zgObMtCmg0hRh9VupT6mGY5smKKjGovhDXtFc6vZ6iUf1H58AeV6W+c12/Bqq/BdF0NyKAzCGmKqUAYxQjjFEGUIKLzmPRZKB/Waw6Y2yHeKoyaRdPWwZiObrer+vl9ucTq6QeeiJ+rFdbrNZZUK3Kh1ugoJSGDbtb3u2dYJLfA7BZ026O1ucDdnY/FYoHtdovdbkf8wmazwbfHR0wmk+dkRw7CS5JcKZlu1PZC+QL6zXVdtQXT6RRlWWI2m6kt8H0fIQ3smOzkyoeEUmR5BMkMU9VkO+RayWUfDAZKEASBumIqV+ikd0f+AcBmYOtXGEYsAAAAAElFTkSuQmCC')",backgroundSize:"cover",display:"block"}}),"\n  ",Object(b.b)("img",{parentName:"span",className:"gatsby-resp-image-image",alt:"Grid Example",title:"Grid Example",src:"/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png",srcSet:["/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/7fc1e/Article_05.png 288w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/a5df1/Article_05.png 576w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png 1152w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/362ee/Article_05.png 1600w"],sizes:"(max-width: 1152px) 100vw, 1152px",style:{width:"100%",height:"100%",margin:"0",verticalAlign:"middle",position:"absolute",top:"0",left:"0"},loading:"lazy"}),"\n    ")),Object(b.b)(d,{colMd:4,colLg:4,noGutterSm:!0,mdxType:"Column"},Object(b.b)("span",{className:"gatsby-resp-image-wrapper",style:{position:"relative",display:"block",marginLeft:"auto",marginRight:"auto",maxWidth:"1152px"}},"\n      ",Object(b.b)("span",{parentName:"span",className:"gatsby-resp-image-background-image",style:{paddingBottom:"56.25%",position:"relative",bottom:"0",left:"0",backgroundImage:"url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAALCAYAAAB/Ca1DAAAACXBIWXMAAAsTAAALEwEAmpwYAAACE0lEQVQoz1WSS2/aQBRGs2DdKsFj4xcBE5FSERahjfFzPDavmqpqEFLaZUMbKrFmgUT/VaUu+uO+3hkgJYujgYs5/ubee8aYhnrzGu0bH1rNBTMvoVsNQp5NaM4bWLaDKAyQxLEiiiIUeY6vXx4wn9+rWnzgjDGGRqOB3k0XjuPCMAxUq1VomqaoVi9gmibihIOLERKeEQJ5Mcb9fIFPnxdIklTJkiTZCz3PQ6/XQ6fTQbvdVt8VrRaazSYsyyRJgWxY7qUpJzKkYgxelPT5RChTSElRDMGzDELkyOk6QgiqFeCcU3KbEmbqj1LE8/FeTIKU5+oFJ0KGq5YHwWNkaaSQEol8IAhDODYJSSSvpoQk+y8ULxPKoTS8NvrRBLfBSPHuvY9+v6966zgObMtCmg0hRh9VupT6mGY5smKKjGovhDXtFc6vZ6iUf1H58AeV6W+c12/Bqq/BdF0NyKAzCGmKqUAYxQjjFEGUIKLzmPRZKB/Waw6Y2yHeKoyaRdPWwZiObrer+vl9ucTq6QeeiJ+rFdbrNZZUK3Kh1ugoJSGDbtb3u2dYJLfA7BZ026O1ucDdnY/FYoHtdovdbkf8wmazwbfHR0wmk+dkRw7CS5JcKZlu1PZC+QL6zXVdtQXT6RRlWWI2m6kt8H0fIQ3smOzkyoeEUmR5BMkMU9VkO+RayWUfDAZKEASBumIqV+ikd0f+AcBmYOtXGEYsAAAAAElFTkSuQmCC')",backgroundSize:"cover",display:"block"}}),"\n  ",Object(b.b)("img",{parentName:"span",className:"gatsby-resp-image-image",alt:"Grid Example",title:"Grid Example",src:"/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png",srcSet:["/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/7fc1e/Article_05.png 288w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/a5df1/Article_05.png 576w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png 1152w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/362ee/Article_05.png 1600w"],sizes:"(max-width: 1152px) 100vw, 1152px",style:{width:"100%",height:"100%",margin:"0",verticalAlign:"middle",position:"absolute",top:"0",left:"0"},loading:"lazy"}),"\n    "))),Object(b.b)(o,{mdxType:"Title"},"Offset"),Object(b.b)(s,{mdxType:"Row"},Object(b.b)(d,{colMd:4,colLg:4,offsetLg:4,mdxType:"Column"},Object(b.b)("span",{className:"gatsby-resp-image-wrapper",style:{position:"relative",display:"block",marginLeft:"auto",marginRight:"auto",maxWidth:"1152px"}},"\n      ",Object(b.b)("span",{parentName:"span",className:"gatsby-resp-image-background-image",style:{paddingBottom:"56.25%",position:"relative",bottom:"0",left:"0",backgroundImage:"url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAALCAYAAAB/Ca1DAAAACXBIWXMAAAsTAAALEwEAmpwYAAACE0lEQVQoz1WSS2/aQBRGs2DdKsFj4xcBE5FSERahjfFzPDavmqpqEFLaZUMbKrFmgUT/VaUu+uO+3hkgJYujgYs5/ubee8aYhnrzGu0bH1rNBTMvoVsNQp5NaM4bWLaDKAyQxLEiiiIUeY6vXx4wn9+rWnzgjDGGRqOB3k0XjuPCMAxUq1VomqaoVi9gmibihIOLERKeEQJ5Mcb9fIFPnxdIklTJkiTZCz3PQ6/XQ6fTQbvdVt8VrRaazSYsyyRJgWxY7qUpJzKkYgxelPT5RChTSElRDMGzDELkyOk6QgiqFeCcU3KbEmbqj1LE8/FeTIKU5+oFJ0KGq5YHwWNkaaSQEol8IAhDODYJSSSvpoQk+y8ULxPKoTS8NvrRBLfBSPHuvY9+v6966zgObMtCmg0hRh9VupT6mGY5smKKjGovhDXtFc6vZ6iUf1H58AeV6W+c12/Bqq/BdF0NyKAzCGmKqUAYxQjjFEGUIKLzmPRZKB/Waw6Y2yHeKoyaRdPWwZiObrer+vl9ucTq6QeeiJ+rFdbrNZZUK3Kh1ugoJSGDbtb3u2dYJLfA7BZ026O1ucDdnY/FYoHtdovdbkf8wmazwbfHR0wmk+dkRw7CS5JcKZlu1PZC+QL6zXVdtQXT6RRlWWI2m6kt8H0fIQ3smOzkyoeEUmR5BMkMU9VkO+RayWUfDAZKEASBumIqV+ikd0f+AcBmYOtXGEYsAAAAAElFTkSuQmCC')",backgroundSize:"cover",display:"block"}}),"\n  ",Object(b.b)("img",{parentName:"span",className:"gatsby-resp-image-image",alt:"Grid Example",title:"Grid Example",src:"/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png",srcSet:["/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/7fc1e/Article_05.png 288w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/a5df1/Article_05.png 576w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png 1152w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/362ee/Article_05.png 1600w"],sizes:"(max-width: 1152px) 100vw, 1152px",style:{width:"100%",height:"100%",margin:"0",verticalAlign:"middle",position:"absolute",top:"0",left:"0"},loading:"lazy"}),"\n    ")),Object(b.b)(d,{colMd:4,colLg:4,mdxType:"Column"},Object(b.b)("span",{className:"gatsby-resp-image-wrapper",style:{position:"relative",display:"block",marginLeft:"auto",marginRight:"auto",maxWidth:"1152px"}},"\n      ",Object(b.b)("span",{parentName:"span",className:"gatsby-resp-image-background-image",style:{paddingBottom:"56.25%",position:"relative",bottom:"0",left:"0",backgroundImage:"url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAALCAYAAAB/Ca1DAAAACXBIWXMAAAsTAAALEwEAmpwYAAACE0lEQVQoz1WSS2/aQBRGs2DdKsFj4xcBE5FSERahjfFzPDavmqpqEFLaZUMbKrFmgUT/VaUu+uO+3hkgJYujgYs5/ubee8aYhnrzGu0bH1rNBTMvoVsNQp5NaM4bWLaDKAyQxLEiiiIUeY6vXx4wn9+rWnzgjDGGRqOB3k0XjuPCMAxUq1VomqaoVi9gmibihIOLERKeEQJ5Mcb9fIFPnxdIklTJkiTZCz3PQ6/XQ6fTQbvdVt8VrRaazSYsyyRJgWxY7qUpJzKkYgxelPT5RChTSElRDMGzDELkyOk6QgiqFeCcU3KbEmbqj1LE8/FeTIKU5+oFJ0KGq5YHwWNkaaSQEol8IAhDODYJSSSvpoQk+y8ULxPKoTS8NvrRBLfBSPHuvY9+v6966zgObMtCmg0hRh9VupT6mGY5smKKjGovhDXtFc6vZ6iUf1H58AeV6W+c12/Bqq/BdF0NyKAzCGmKqUAYxQjjFEGUIKLzmPRZKB/Waw6Y2yHeKoyaRdPWwZiObrer+vl9ucTq6QeeiJ+rFdbrNZZUK3Kh1ugoJSGDbtb3u2dYJLfA7BZ026O1ucDdnY/FYoHtdovdbkf8wmazwbfHR0wmk+dkRw7CS5JcKZlu1PZC+QL6zXVdtQXT6RRlWWI2m6kt8H0fIQ3smOzkyoeEUmR5BMkMU9VkO+RayWUfDAZKEASBumIqV+ikd0f+AcBmYOtXGEYsAAAAAElFTkSuQmCC')",backgroundSize:"cover",display:"block"}}),"\n  ",Object(b.b)("img",{parentName:"span",className:"gatsby-resp-image-image",alt:"Grid Example",title:"Grid Example",src:"/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png",srcSet:["/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/7fc1e/Article_05.png 288w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/a5df1/Article_05.png 576w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/3cbba/Article_05.png 1152w","/mcas/static/dc51d23a5322c2511205c8c525bbe8ee/362ee/Article_05.png 1600w"],sizes:"(max-width: 1152px) 100vw, 1152px",style:{width:"100%",height:"100%",margin:"0",verticalAlign:"middle",position:"absolute",top:"0",left:"0"},loading:"lazy"}),"\n    "))),Object(b.b)("h3",null,"Code"),Object(b.b)("pre",null,Object(b.b)("code",{parentName:"pre",className:"language-jsx",metastring:"path=components/Grid.js src=https://github.com/carbon-design-system/gatsby-theme-carbon/tree/master/packages/gatsby-theme-carbon/src/components/Grid",path:"components/Grid.js",src:"https://github.com/carbon-design-system/gatsby-theme-carbon/tree/master/packages/gatsby-theme-carbon/src/components/Grid"},"<Row>\n  <Column colMd={4} colLg={4}>\n    ![Grid Example](images/Article_05.png)\n  </Column>\n  <Column colMd={4} colLg={4}>\n    ![Grid Example](images/Article_05.png)\n  </Column>\n  <Column colMd={4} colLg={4}>\n    ![Grid Example](images/Article_05.png)\n  </Column>\n</Row>\n")),Object(b.b)(o,{mdxType:"Title"},"No gutter left"),Object(b.b)("pre",null,Object(b.b)("code",{parentName:"pre",className:"language-jsx",metastring:"path=components/Grid.js src=https://github.com/carbon-design-system/gatsby-theme-carbon/tree/master/packages/gatsby-theme-carbon/src/components/Grid",path:"components/Grid.js",src:"https://github.com/carbon-design-system/gatsby-theme-carbon/tree/master/packages/gatsby-theme-carbon/src/components/Grid"},"<Row>\n  <Column colMd={4} colLg={4} noGutterMdLeft>\n    ![Grid Example](images/Article_05.png)\n  </Column>\n  <Column colMd={4} colLg={4} noGutterMdLeft>\n    ![Grid Example](images/Article_05.png)\n  </Column>\n  <Column colMd={4} colLg={4} noGutterMdLeft>\n    ![Grid Example](images/Article_05.png)\n  </Column>\n</Row>\n")),Object(b.b)(o,{mdxType:"Title"},"No gutter"),Object(b.b)("pre",null,Object(b.b)("code",{parentName:"pre",className:"language-jsx",metastring:"path=components/Grid.js src=https://github.com/carbon-design-system/gatsby-theme-carbon/tree/master/packages/gatsby-theme-carbon/src/components/Grid",path:"components/Grid.js",src:"https://github.com/carbon-design-system/gatsby-theme-carbon/tree/master/packages/gatsby-theme-carbon/src/components/Grid"},"<Row>\n  <Column colMd={4} colLg={4} noGutterSm>\n    ![Grid Example](images/Article_05.png)\n  </Column>\n  <Column colMd={4} colLg={4} noGutterSm>\n    ![Grid Example](images/Article_05.png)\n  </Column>\n  <Column colMd={4} colLg={4} noGutterSm>\n    ![Grid Example](images/Article_05.png)\n  </Column>\n</Row>\n")),Object(b.b)(o,{mdxType:"Title"},"Offset"),Object(b.b)("pre",null,Object(b.b)("code",{parentName:"pre",className:"language-jsx",metastring:"path=components/Grid.js src=https://github.com/carbon-design-system/gatsby-theme-carbon/tree/master/packages/gatsby-theme-carbon/src/components/Grid",path:"components/Grid.js",src:"https://github.com/carbon-design-system/gatsby-theme-carbon/tree/master/packages/gatsby-theme-carbon/src/components/Grid"},"<Row>\n  <Column colMd={4} colLg={4} offsetLg={4}>\n    ![Grid Example](images/Article_05.png)\n  </Column>\n  <Column colMd={4} colLg={4}>\n    ![Grid Example](images/Article_05.png)\n  </Column>\n</Row>\n")),Object(b.b)(o,{mdxType:"Title"},"Column props"),Object(b.b)("table",null,Object(b.b)("thead",{parentName:"table"},Object(b.b)("tr",{parentName:"thead"},Object(b.b)("th",{parentName:"tr",align:null},"property"),Object(b.b)("th",{parentName:"tr",align:null},"propType"),Object(b.b)("th",{parentName:"tr",align:null},"required"),Object(b.b)("th",{parentName:"tr",align:null},"default"),Object(b.b)("th",{parentName:"tr",align:null},"description"))),Object(b.b)("tbody",{parentName:"table"},Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"children"),Object(b.b)("td",{parentName:"tr",align:null},"node"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null})),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"className"),Object(b.b)("td",{parentName:"tr",align:null},"string"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Add custom class name")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"colSm"),Object(b.b)("td",{parentName:"tr",align:null},"number"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify the col width at small breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"colMd"),Object(b.b)("td",{parentName:"tr",align:null},"number"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify the col width at medium breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"colLg"),Object(b.b)("td",{parentName:"tr",align:null},"number"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"12"),Object(b.b)("td",{parentName:"tr",align:null},"Specify the col width at large breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"colXl"),Object(b.b)("td",{parentName:"tr",align:null},"number"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify the col width at x-large breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"colMax"),Object(b.b)("td",{parentName:"tr",align:null},"number"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify the col width at max breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"offsetSm"),Object(b.b)("td",{parentName:"tr",align:null},"number"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify the col offset at small breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"offsetMd"),Object(b.b)("td",{parentName:"tr",align:null},"number"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify the col offset at medium breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"offsetLg"),Object(b.b)("td",{parentName:"tr",align:null},"number"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify the col offset at large breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"offsetXl"),Object(b.b)("td",{parentName:"tr",align:null},"number"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify the col offset at x-large breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"offsetMax"),Object(b.b)("td",{parentName:"tr",align:null},"number"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify the col offset at max breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"noGutterSm"),Object(b.b)("td",{parentName:"tr",align:null},"bool"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify no-gutter at small breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"noGutterMd"),Object(b.b)("td",{parentName:"tr",align:null},"bool"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify no-gutter at medium breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"noGutterLg"),Object(b.b)("td",{parentName:"tr",align:null},"bool"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify no-gutter at large breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"noGutterXl"),Object(b.b)("td",{parentName:"tr",align:null},"bool"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify no-gutter at x-large breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"noGutterMax"),Object(b.b)("td",{parentName:"tr",align:null},"bool"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify no-gutter at max breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"noGutterSmLeft"),Object(b.b)("td",{parentName:"tr",align:null},"bool"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify no-gutter left at small breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"noGutterMdLeft"),Object(b.b)("td",{parentName:"tr",align:null},"bool"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify no-gutter left at medium breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"noGutterLgLeft"),Object(b.b)("td",{parentName:"tr",align:null},"bool"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify no-gutter left at large breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"noGutterXlLeft"),Object(b.b)("td",{parentName:"tr",align:null},"bool"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify no-gutter left at x-large breakpoint")),Object(b.b)("tr",{parentName:"tbody"},Object(b.b)("td",{parentName:"tr",align:null},"noGutterMaxLeft"),Object(b.b)("td",{parentName:"tr",align:null},"bool"),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null}),Object(b.b)("td",{parentName:"tr",align:null},"Specify no-gutter left at max breakpoint")))))}u.isMDXComponent=!0},QH2O:function(e,t,a){e.exports={bxTextTruncateEnd:"PageHeader-module--bx--text-truncate--end--mZWeX",bxTextTruncateFront:"PageHeader-module--bx--text-truncate--front--3zvrI",pageHeader:"PageHeader-module--page-header--3hIan",darkMode:"PageHeader-module--dark-mode--hBrwL",withTabs:"PageHeader-module--with-tabs--3nKxA",text:"PageHeader-module--text--o9LFq"}}}]);
//# sourceMappingURL=component---src-pages-components-grid-mdx-6e727eb101e4ebb854ae.js.map