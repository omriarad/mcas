(window.webpackJsonp=window.webpackJsonp||[]).push([[40],{"013z":function(e,t,a){"use strict";var n=a("q1tI"),r=a.n(n),o=a("NmYn"),l=a.n(o),b=a("Wbzz"),i=a("Xrax"),c=a("k4MR"),s=a("TSYQ"),p=a.n(s),m=a("QH2O"),d=a.n(m),u=a("qKvR"),O=function(e){var t,a=e.title,n=e.theme,r=e.tabs,o=void 0===r?[]:r;return Object(u.b)("div",{className:p()(d.a.pageHeader,(t={},t[d.a.withTabs]=o.length,t[d.a.darkMode]="dark"===n,t))},Object(u.b)("div",{className:"bx--grid"},Object(u.b)("div",{className:"bx--row"},Object(u.b)("div",{className:"bx--col-lg-12"},Object(u.b)("h1",{id:"page-title",className:d.a.text},a)))))},j=a("BAC9"),h=function(e){var t=e.relativePagePath,a=e.repository,n=Object(b.useStaticQuery)("1364590287").site.siteMetadata.repository,r=a||n,o=r.baseUrl,l=r.subDirectory,i=o+"/edit/"+r.branch+l+"/src/pages"+t;return o?Object(u.b)("div",{className:"bx--row "+j.row},Object(u.b)("div",{className:"bx--col"},Object(u.b)("a",{className:j.link,href:i},"Edit this page on GitHub"))):null},g=a("FCXl"),N=a("dI71"),f=a("I8xM"),x=function(e){function t(){return e.apply(this,arguments)||this}return Object(N.a)(t,e),t.prototype.render=function(){var e=this.props,t=e.title,a=e.tabs,n=e.slug,r=n.split("/").filter(Boolean).slice(-1)[0],o=a.map((function(e){var t,a=l()(e,{lower:!0,strict:!0}),o=a===r,i=new RegExp(r+"/?(#.*)?$"),c=n.replace(i,a);return Object(u.b)("li",{key:e,className:p()((t={},t[f.selectedItem]=o,t),f.listItem)},Object(u.b)(b.Link,{className:f.link,to:""+c},e))}));return Object(u.b)("div",{className:f.tabsContainer},Object(u.b)("div",{className:"bx--grid"},Object(u.b)("div",{className:"bx--row"},Object(u.b)("div",{className:"bx--col-lg-12 bx--col-no-gutter"},Object(u.b)("nav",{"aria-label":t},Object(u.b)("ul",{className:f.list},o))))))},t}(r.a.Component),y=a("MjG9"),v=a("CzIb");t.a=function(e){var t=e.pageContext,a=e.children,n=e.location,r=e.Title,o=t.frontmatter,s=void 0===o?{}:o,p=t.relativePagePath,m=t.titleType,d=s.tabs,j=s.title,N=s.theme,f=s.description,w=s.keywords,T=Object(v.a)().interiorTheme,k=Object(b.useStaticQuery)("2456312558").site.pathPrefix,P=k?n.pathname.replace(k,""):n.pathname,M=d?P.split("/").filter(Boolean).slice(-1)[0]||l()(d[0],{lower:!0}):"",C=N||T;return Object(u.b)(c.a,{tabs:d,homepage:!1,theme:C,pageTitle:j,pageDescription:f,pageKeywords:w,titleType:m},Object(u.b)(O,{title:r?Object(u.b)(r,null):j,label:"label",tabs:d,theme:C}),d&&Object(u.b)(x,{title:j,slug:P,tabs:d,currentTab:M}),Object(u.b)(y.a,{padded:!0},a,Object(u.b)(h,{relativePagePath:p})),Object(u.b)(g.a,{pageContext:t,location:n,slug:P,tabs:d,currentTab:M}),Object(u.b)(i.a,null))}},BAC9:function(e,t,a){e.exports={bxTextTruncateEnd:"EditLink-module--bx--text-truncate--end--2pqje",bxTextTruncateFront:"EditLink-module--bx--text-truncate--front--3_lIE",link:"EditLink-module--link--1qzW3",row:"EditLink-module--row--1B9Gk"}},I8xM:function(e,t,a){e.exports={bxTextTruncateEnd:"PageTabs-module--bx--text-truncate--end--267NA",bxTextTruncateFront:"PageTabs-module--bx--text-truncate--front--3xEQF",tabsContainer:"PageTabs-module--tabs-container--8N4k0",list:"PageTabs-module--list--3eFQc",listItem:"PageTabs-module--list-item--nUmtD",link:"PageTabs-module--link--1mDJ1",selectedItem:"PageTabs-module--selected-item--YPVr3"}},QH2O:function(e,t,a){e.exports={bxTextTruncateEnd:"PageHeader-module--bx--text-truncate--end--mZWeX",bxTextTruncateFront:"PageHeader-module--bx--text-truncate--front--3zvrI",pageHeader:"PageHeader-module--page-header--3hIan",darkMode:"PageHeader-module--dark-mode--hBrwL",withTabs:"PageHeader-module--with-tabs--3nKxA",text:"PageHeader-module--text--o9LFq"}},Uxid:function(e,t,a){"use strict";a.r(t),a.d(t,"_frontmatter",(function(){return i})),a.d(t,"default",(function(){return m}));var n,r=a("wx14"),o=a("zLVn"),l=(a("q1tI"),a("7ljp")),b=a("013z"),i=(a("qKvR"),{}),c=(n="PageDescription",function(e){return console.warn("Component "+n+" was not imported, exported, or provided by MDXProvider as global scope"),Object(l.b)("div",e)}),s={_frontmatter:i},p=b.a;function m(e){var t=e.components,a=Object(o.a)(e,["components"]);return Object(l.b)(p,Object(r.a)({},s,a,{components:t,mdxType:"MDXLayout"}),Object(l.b)(c,{mdxType:"PageDescription"},Object(l.b)("p",null,"MCAS is designed for flexible deployment.  However, it is positioned for\nspecific network and memory hardware if available.")),Object(l.b)("h2",null,"Platform Preparation"),Object(l.b)("ol",null,Object(l.b)("li",{parentName:"ol"},Object(l.b)("p",{parentName:"li"},Object(l.b)("strong",{parentName:"p"},"Operating System")," - install one of the following supported distributions:"),Object(l.b)("ul",{parentName:"li"},Object(l.b)("li",{parentName:"ul"},"Ubuntu 18.04 LTS x86_64"),Object(l.b)("li",{parentName:"ul"},"Fedora Core 27, 30 or 32 x86_64"),Object(l.b)("li",{parentName:"ul"},"RHEL8 x86_64"))),Object(l.b)("li",{parentName:"ol"},Object(l.b)("p",{parentName:"li"},Object(l.b)("strong",{parentName:"p"},"Mellanox RDMA")," - for high-performance MCAS supports Mellanox RDMA network\ncards and has been tested with ConnectX-4 and ConnectX-5.  MCAS can operate with\nplain TCP/IP sockets, but performance is significantly slower.  Mellanox OFED\n(OpenFabrics Enterprise Distribution for Linux) distributions can be downloaded\nfrom ",Object(l.b)("a",{parentName:"p",href:"https://www.mellanox.com/products/infiniband-drivers/linux/mlnx_ofed"},"https://www.mellanox.com/products/infiniband-drivers/linux/mlnx_ofed"),"."),Object(l.b)("pre",{parentName:"li"},Object(l.b)("code",{parentName:"pre",className:"language-bash"},"  $ ibdev2netdev\n  mlx5_0 port 1 ==> enp216s0f0 (Up)\n  mlx5_1 port 1 ==> enp216s0f1 (Up)\n"))),Object(l.b)("li",{parentName:"ol"},Object(l.b)("p",{parentName:"li"},Object(l.b)("strong",{parentName:"p"},"Persistent Memory")," - MCAS is designed explicitly for persistent memory.  However,\nit can be used with DRAM only (mapstore) or with emulated persistent memory.\nEmulated or real persistent memory must be\nconfigured in ",Object(l.b)("em",{parentName:"p"},"device DAX")," mode.  See ",Object(l.b)("a",{parentName:"p",href:"https://pmem.io/2016/02/22/pm-emulation.html"},"https://pmem.io/2016/02/22/pm-emulation.html")," for\ninformation on PM emulation.  Verify availability of devdax PM:"),Object(l.b)("pre",{parentName:"li"},Object(l.b)("code",{parentName:"pre",className:"language-bash"},"ls /dev/dax*\nchmod a+rwx /dev/dax*\n")))),Object(l.b)("h2",null,"Building MCAS"),Object(l.b)("ol",null,Object(l.b)("li",{parentName:"ol"},Object(l.b)("p",{parentName:"li"},"Check out source and update submodules:"),Object(l.b)("pre",{parentName:"li"},Object(l.b)("code",{parentName:"pre",className:"language-sh"},"git clone https://github.com/IBM/mcas.git\ncd mcas\ngit submodule update --init --recursive\n"))),Object(l.b)("li",{parentName:"ol"},Object(l.b)("p",{parentName:"li"},"Install package dependencies. For example:"),Object(l.b)("pre",{parentName:"li"},Object(l.b)("code",{parentName:"pre",className:"language-sh"},"cd mcas/deps\nsudo ./install-yum-fc27.sh\n"))),Object(l.b)("li",{parentName:"ol"},Object(l.b)("p",{parentName:"li"},"Configure cmake build (e.g. release build):"),Object(l.b)("pre",{parentName:"li"},Object(l.b)("code",{parentName:"pre",className:"language-sh"},"mkdir build\ncd build\ncmake -DBUILD_KERNEL_SUPPORT=ON -DFLATBUFFERS_BUILD_TESTS=0 -DTBB_BUILD_TESTS=0 -DBUILD_PYTHON_SUPPORT=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/dist ..\n"))),Object(l.b)("li",{parentName:"ol"},Object(l.b)("p",{parentName:"li"},"Make bootstrap (this only needs to happen once):"),Object(l.b)("pre",{parentName:"li"},Object(l.b)("code",{parentName:"pre",className:"language-sh"},"make bootstrap\n"))),Object(l.b)("li",{parentName:"ol"},Object(l.b)("p",{parentName:"li"},"Perform rest of build (which can be repeated on code change):"),Object(l.b)("pre",{parentName:"li"},Object(l.b)("code",{parentName:"pre",className:"language-sh"},"make -j install\n")))),Object(l.b)("h2",null,"Before running code"),Object(l.b)("p",null,"MCAS currently requires two custom kernel modules.  One, ",Object(l.b)("inlineCode",{parentName:"p"},"xpmem.ko")," is needed for the ‘mapstore’ backend component.  The other, ",Object(l.b)("inlineCode",{parentName:"p"},"mcasmod.ko")," is needed for the ‘hstore’ components.  Normmaly, both modules should be loaded into the system after they have been build."),Object(l.b)("pre",null,Object(l.b)("code",{parentName:"pre",className:"language-sh"},"insmod ./dist/lib/modules/4.18.19-100.fc27.x86_64/xpmem.ko\n")),Object(l.b)("pre",null,Object(l.b)("code",{parentName:"pre",className:"language-sh"},"insmod ./dist/bin/mcasmod.ko\n")),Object(l.b)("p",null,Object(l.b)("strong",{parentName:"p"},"Note"),": the ",Object(l.b)("em",{parentName:"p"},"hstore")," backend is preferred.  If you are using the\n",Object(l.b)("em",{parentName:"p"},"mapstore")," backend, the direct operations (e.g. ‘get_direct,\nput_direct) that perform zero-copy DMA transfers will not work."),Object(l.b)("h2",null,"Running an example"),Object(l.b)("h3",null,"Launch MCAS server"),Object(l.b)("p",null,"The MCAS server can be launched from the build directory.  Using one of the pre-supplied (testing) configuration files:"),Object(l.b)("pre",null,Object(l.b)("code",{parentName:"pre",className:"language-bash"},"./dist/bin/mcas --conf ./dist/testing/mapstore-0.conf\n")),Object(l.b)("p",null,"This configuration file defines a single shard, using port 11911 on the ",Object(l.b)("inlineCode",{parentName:"p"},"mlx5_0")," RDMA NIC adapter."),Object(l.b)("p",null,"Note, ",Object(l.b)("inlineCode",{parentName:"p"},"./dist")," is the location of the installed distribution."),Object(l.b)("h3",null,"Launch the Python client"),Object(l.b)("p",null,"Again, from the build directory:"),Object(l.b)("pre",null,Object(l.b)("code",{parentName:"pre",className:"language-bash"},"./dist/bin/mcas-shell\n")),Object(l.b)("p",null,"First open a session to the MCAS server:"),Object(l.b)("pre",null,Object(l.b)("code",{parentName:"pre",className:"language-python"},"session = mcas.Session(ip='10.0.0.101', port=11911)\n")),Object(l.b)("p",null,"Next create a pool. Provide pool name, size of pool in bytes and expected number of objects (presizes hash table):"),Object(l.b)("pre",null,Object(l.b)("code",{parentName:"pre",className:"language-python"},"pool = session.create_pool('pool0', 64*1024, 1000)\n")),Object(l.b)("p",null,"Now we can create key-value pairs:"),Object(l.b)("pre",null,Object(l.b)("code",{parentName:"pre",className:"language-python"},"pool.put('myPet','doggy')\n")),Object(l.b)("p",null,"And then retrieve the value back:"),Object(l.b)("pre",null,Object(l.b)("code",{parentName:"pre",className:"language-python"},"pool.get('myPet')\n")),Object(l.b)("p",null,"We can configure a volatile index for the pool.  This allows us to perform scans on the key space - find_key(expression, offset)."),Object(l.b)("pre",null,Object(l.b)("code",{parentName:"pre",className:"language-python"},"pool.configure(\"AddIndex::VolatileTree\")\npool.find_key('regex:.*', 0)\n")),Object(l.b)("p",null,"Finally, the pool can be closed."),Object(l.b)("pre",null,Object(l.b)("code",{parentName:"pre",className:"language-python"},"pool.close()\n")))}m.isMDXComponent=!0}}]);
//# sourceMappingURL=component---src-pages-getting-started-index-mdx-26b51f3c091ea4bde041.js.map